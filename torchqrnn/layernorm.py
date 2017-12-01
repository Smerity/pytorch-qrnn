import math

import torch
from torch import nn
from torch.autograd import Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


kernel = '''
extern "C"
__global__ void layer_norm(float *dst, float *nonscaleddst, float *resstd, float *resmean, const float *x, const float *gamma, const float *beta, int SEQ, int BATCH, int HIDDEN)
{
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  int ts = blockIdx.z * blockDim.z + threadIdx.z;
  if(ts >= SEQ || bid >= BATCH || hid >= HIDDEN)
     return;
  //
  int stateidx = ts * BATCH + bid;
  int i = ts * HIDDEN * BATCH + bid * HIDDEN + hid;

  float mean = resmean[stateidx];
  float std = resstd[stateidx];

  nonscaleddst[i] = ((x[i] - mean) / std);
  dst[i] = gamma[hid] * nonscaleddst[i] + beta[hid];
}

extern "C"
__global__ void bwd_layer_norm(const float *h, const float *resstd, const float *resmean, const float *resbotgradh, const float *resmeangrad, const float *x, const float *gamma, const float *beta, const float *gh, float *gx, int SEQ, int BATCH, int HIDDEN)
{
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  int ts = blockIdx.z * blockDim.z + threadIdx.z;
  if(ts >= SEQ || bid >= BATCH || hid >= HIDDEN)
     return;
  //
  int stateidx = ts * BATCH + bid;
  int i = ts * HIDDEN * BATCH + bid * HIDDEN + hid;

  float botgradh = resbotgradh[stateidx];
  float meangrad = resmeangrad[stateidx];
  float std = resstd[stateidx];

  float dx = 2 * (x[i] - resmean[stateidx]) * botgradh;
  dx += (gamma[hid] * gh[i] / std) - meangrad;
  gx[i] = dx;
}

extern "C"
__global__ void bwd_multiplications(const float *non_scaled_result, const float *resstd, const float *resmean, float *grad_gamma, float *botgradh, float *resmeangrad, float *dmean_dx, const float *x, const float *gamma, const float *beta, const float *gh, int SEQ, int BATCH, int HIDDEN)
{
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  int ts = blockIdx.z * blockDim.z + threadIdx.z;
  if(ts >= SEQ || bid >= BATCH || hid >= HIDDEN)
     return;
  //
  int stateidx = ts * BATCH + bid;
  int i = ts * HIDDEN * BATCH + bid * HIDDEN + hid;

  float mean = resmean[stateidx];
  float std = resstd[stateidx];

  /*
  gamma_grad_result = gamma * grad_result.data
  grad_gamma = (self.non_scaled_result * grad_result.data).view(-1, hidden_size).sum(dim=0)
  grad_beta = grad_result.view(-1, hidden_size).sum(dim=0).data
  botgradh = (gamma_grad_result * ((self.resmean - x) / (self.resstd ** 2))).sum(dim=-1, keepdim=True) / (2 * self.resstd) / (hidden_size - 1)
  resmeangrad = (gamma_grad_result / self.resstd).sum(dim=-1, keepdim=True) / hidden_size
  dmean_dx = (2 * (-1) * (1 - (1 / hidden_size))) * (x - self.resmean).sum(dim=-1, keepdim=True) * botgradh
  */

  float gamma_grad_result = gamma[hid] * gh[i];
  grad_gamma[i] = non_scaled_result[i] * gh[i];
  botgradh[i] = gamma_grad_result * ((mean - x[i]) / powf(std, 2));
  resmeangrad[i] = gamma_grad_result / std;
  dmean_dx[i] = x[i] - mean;
}

'''


class GPULayerNorm(torch.autograd.Function):
    configured_gpus = {}
    ptx = None

    @staticmethod
    def compile(self):
        if GPULayerNorm.ptx is None:
            program = Program(kernel.encode(), 'layer_norm.cu'.encode())
            GPULayerNorm.ptx = program.compile()

        if torch.cuda.current_device() not in GPULayerNorm.configured_gpus:
            m = function.Module()
            m.load(bytes(GPULayerNorm.ptx.encode()))

            self.layer_norm = m.get_function('layer_norm')
            self.bwd_layer_norm = m.get_function('bwd_layer_norm')
            self.bwd_mults = m.get_function('bwd_multiplications')

            Stream = namedtuple('Stream', ['ptr'])
            self.stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

            GPULayerNorm.configured_gpus[torch.cuda.current_device()] = (self.layer_norm, self.bwd_layer_norm, self.bwd_mults, self.stream)

        self.layer_norm, self.bwd_layer_norm, self.bwd_mults, self.stream = GPULayerNorm.configured_gpus[torch.cuda.current_device()]

    @staticmethod
    def forward(self, x, gamma, beta, eps=1e-6):
        GPULayerNorm.compile(self)
        assert x.is_contiguous(), 'Inputs must be contiguous for GPU kernel'
        seq_size, batch_size, hidden_size = x.size()
        self.non_scaled_result = x.new(seq_size, batch_size, hidden_size)
        result = x.new(seq_size, batch_size, hidden_size)
        self.resstd = (x.std(-1, keepdim=True) + eps).contiguous()
        self.resmean = x.mean(-1, keepdim=True).contiguous()
        ###
        # Block is limited to (512, 512, 64) for (x, y, z)
        # https://stackoverflow.com/questions/5062781/cuda-max-threads-in-a-block
        # Grid specifies full required space
        hid, bid, ts = 256, 1, 1
        grid_hidden_size = min(hidden_size, hid)
        grid = (math.ceil(hidden_size / grid_hidden_size), math.ceil(batch_size / bid), math.ceil(seq_size / ts))
        self.layer_norm(grid=grid, block=(grid_hidden_size, bid, ts), args=[result.data_ptr(), self.non_scaled_result.data_ptr(), self.resstd.data_ptr(), self.resmean.data_ptr(), x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), seq_size, batch_size, hidden_size], stream=self.stream)
        self.save_for_backward(result, x, gamma, beta)
        return result

    @staticmethod
    def backward(self, grad_result):
        GPULayerNorm.compile(self)
        # There is no guarantee the gradient tensor is contiguous
        grad_result = grad_result.contiguous()
        result, x, gamma, beta = self.saved_tensors
        ###
        seq_size, batch_size, hidden_size = x.size()
        # Zeroing of grad_x is done via the CUDA kernel (overwrites the memory)
        grad_x = x.new(*x.size())
        ###
        grad_beta = grad_result.view(-1, hidden_size).sum(dim=0).data
        grad_gamma = x.new(*x.size())
        botgradh = x.new(*x.size())
        resmeangrad = x.new(*x.size())
        dmean_dx = x.new(*x.size())
        ###
        hid, bid, ts = 256, 1, 1
        grid_hidden_size = min(hidden_size, hid)
        grid = (math.ceil(hidden_size / grid_hidden_size), math.ceil(batch_size / bid), math.ceil(seq_size / ts))
        ###
        self.bwd_mults(grid=grid, block=(grid_hidden_size, bid, ts), args=[self.non_scaled_result.data_ptr(), self.resstd.data_ptr(), self.resmean.data_ptr(), grad_gamma.data_ptr(), botgradh.data_ptr(), resmeangrad.data_ptr(), dmean_dx.data_ptr(), x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), grad_result.data.data_ptr(), seq_size, batch_size, hidden_size], stream=self.stream)
        ###
        botgradh = botgradh.sum(dim=-1, keepdim=True) / (2 * self.resstd) / (hidden_size - 1)
        resmeangrad = resmeangrad.sum(dim=-1, keepdim=True) / hidden_size
        dmean_dx = dmean_dx.sum(dim=-1, keepdim=True) * botgradh
        grad_gamma = grad_gamma.view(-1, hidden_size).sum(dim=0)
        ###
        self.bwd_layer_norm(grid=grid, block=(grid_hidden_size, bid, ts), args=[result.data_ptr(), self.resstd.data_ptr(), self.resmean.data_ptr(), botgradh.data_ptr(), resmeangrad.data_ptr(), x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), grad_result.data.data_ptr(), grad_x.data_ptr(), seq_size, batch_size, hidden_size], stream=self.stream)
        ###
        grad_x += dmean_dx
        ###
        xgrad, gammagrad, betagrad = (Variable(v, volatile=True) for v in (grad_x, grad_gamma, grad_beta))
        return xgrad, gammagrad, betagrad


class CPULayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.features = features
        self.eps = eps

    def forward(self, x, gamma, beta):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return gamma * ((x - mean) / (std + self.eps)) + beta


class LayerNorm(torch.nn.Module):
    def __init__(self, features, *args, use_custom_cuda=True, **kwargs):
        super().__init__()
        # Use CUDA by default unless it's available
        self.use_custom_cuda = use_custom_cuda and torch.cuda.is_available()
        self.lnorm = GPULayerNorm if self.use_custom_cuda else CPULayerNorm(features, *args, **kwargs)
        if self.use_custom_cuda:
            self.gamma = nn.Parameter(torch.ones(features).cuda())
            self.beta = nn.Parameter(torch.zeros(features).cuda())
        else:
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        if self.use_custom_cuda: x = x.contiguous()
        if self.lnorm == GPULayerNorm: return self.lnorm.apply(x, self.gamma, self.beta)
        return self.lnorm(x, self.gamma, self.beta)

###

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Warning: Layer norm can't work for hidden size = 1 (i.e. (3, 1, 1))
    seq_len, batch_size, hidden_size = 35, 20, 128
    #seq_len, batch_size, hidden_size = 5, 4, 8
    seq_len, batch_size, hidden_size = 3, 3, 3
    #seq_len, batch_size, hidden_size = 3, 2, 4
    size = (seq_len, batch_size, hidden_size)
    x = 10 * torch.rand(size)
    g, b = torch.rand(hidden_size) * 2.5, torch.rand(hidden_size) * 0.8
    #print('x', x)

    cpuX = Variable(x, requires_grad=True)
    cpuln = LayerNorm(hidden_size, use_custom_cuda=False)
    cpuln.gamma.data = g
    cpuln.beta.data = b
    v = cpuln(cpuX)
    cpuloss = (v ** 2).sum()
    cpuloss.backward()
    print('CPU:', cpuloss.data[0])
    print('CPU gradx(sum=1):', cpuX.grad.sum(dim=1).data[0])
    #print('CPU gradx:', cpuX.grad.data)
    print('CPU gradg:', cpuln.gamma.grad.data)
    print('CPU gradb:', cpuln.beta.grad.data)

    print('=-=-=-=-=-=')

    gpuX = Variable(x.cuda(), requires_grad=True)
    gpuln = LayerNorm(hidden_size, use_custom_cuda=True).cuda()
    gpuln.gamma.data = g.cuda()
    gpuln.beta.data = b.cuda()
    # Perform sum on CPU side as GPU sum can be different
    v = gpuln(gpuX).cpu()
    gpuloss = (v ** 2).sum()
    gpuloss.backward()
    print('GPU:', gpuloss.data[0])
    print('GPU gradx(sum=1):', gpuX.grad.sum(dim=1).data[0])
    #print('GPU gradx:', gpuX.grad.data)
    print('GPU gradg:', gpuln.gamma.grad.data)
    print('GPU gradb:', gpuln.beta.grad.data)

import torch
import torch.nn.functional as F
from torch import nn
from .grad_utils import GradientsHelper

def postprocess_darcy(x: torch.Tensor, discretize_a=True):
    # Note that, in reward finetuning, we need discretize_a to be True
    # However, in DiffusionPDE, during guidance, they don't need to discretize a
    # For evaluation, we need to discretize a, so we need to set discretize_a to True
    assert x.shape[1] == 2 and x.ndim == 4
    a, u = x.chunk(2, dim=1)
    a = ((a+1.5)/0.2)
    if discretize_a:
        a[a>7.5] = 12 # a is binary
        a[a<=7.5] = 3
    u = ((u+0.9)/115)
    return torch.cat([a, u], dim=1)

def postprocess_burgers(x: torch.Tensor):
    assert x.shape[1] == 1 and x.ndim == 4
    return x * 1.415

class DarcyResidual(nn.Module):
    def __init__(self, postprocess_input: bool, discretize_a: bool, domain_length = 1., pixels_per_dim = 128, pixels_at_boundary = False, fd_acc = 2, device = 'cpu'):    
        super().__init__()
        # assert data_source in ['diffusionpde', 'pidm']
        self.periodic = False
        self.pixels_at_boundary = pixels_at_boundary
        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim
        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.input_dim = 2
        self.postprocess_input = postprocess_input
        # for guidance, we don't need to discretize a; for final output, we need to discretize a
        self.discretize_a = discretize_a 
        # self.data_source = data_source
        
        # create stationary source field        
        # w = 0.125
        # r = 10.0
        # domain_size = 1.
        # create point grid
        pixel_size = domain_length / pixels_per_dim
        start = pixel_size / 2
        end = domain_length - pixel_size / 2
        x = torch.linspace(start, end, steps=pixels_per_dim)
        y = torch.linspace(start, end, steps=pixels_per_dim)
        X, _ = torch.meshgrid(x, y, indexing='ij')
        # compute the function values on the grid
        f_s = self.create_f_s_diffusionpde(X) 
        self.f_s = f_s.unsqueeze(0).to(device)
    
    @classmethod
    def postprocess_darcy(cls, x: torch.Tensor, discretize_a: bool):
        # Note that, in reward finetuning, we need discretize_a to be True
        # However, in DiffusionPDE, during guidance, they don't need to discretize a
        # For evaluation, we need to discretize a, so we need to set discretize_a to True
        assert x.shape[1] == 2 and x.ndim == 4
        a, u = x.chunk(2, dim=1)
        a = ((a+1.5)/0.2)
        if discretize_a:
            a[a>7.5] = 12 # a is binary
            a[a<=7.5] = 3
        u = ((u+0.9)/115)
        return torch.cat([a, u], dim=1)
        
    def create_f_s_diffusionpde(self, x):
        result = torch.ones_like(x) 
        return result
    
    def forward(self, x0_pred: torch.Tensor):
        assert x0_pred.shape[1] == 2 and x0_pred.ndim == 4
        if self.postprocess_input:
            x0_pred = self.postprocess_darcy(x0_pred, self.discretize_a)
        permeability_field, p = x0_pred.chunk(2, dim=1)
        p_d0 = self.grads.stencil_gradients(p, mode='d_d0')
        p_d1 = self.grads.stencil_gradients(p, mode='d_d1')
        p_d00 = self.grads.stencil_gradients(p, mode='d_d00')
        p_d11 = self.grads.stencil_gradients(p, mode='d_d11')
        perm_d0 = self.grads.stencil_gradients(permeability_field, mode='d_d0')
        perm_d1 = self.grads.stencil_gradients(permeability_field, mode='d_d1')
        v00 = -permeability_field * p_d00 - perm_d0 * p_d0
        v11 = -permeability_field * p_d11 - perm_d1 * p_d1
        residual = v00 + v11 - self.f_s
        return residual


class Darcy64Residual(nn.Module):
    def __init__(self, postprocess_input: bool, domain_length = 1., pixels_per_dim = 64, pixels_at_boundary = False, fd_acc = 2, device = 'cpu', w=0.125, r=10.):    
        super().__init__()
        # assert data_source in ['diffusionpde', 'pidm']
        self.periodic = False
        self.pixels_at_boundary = pixels_at_boundary
        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim
        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.input_dim = 2
        self.postprocess_input = postprocess_input
        
        # create stationary source field        
        pixel_size = domain_length / pixels_per_dim
        start = pixel_size / 2
        end = domain_length - pixel_size / 2
        x = torch.linspace(start, end, steps=pixels_per_dim)
        y = torch.linspace(start, end, steps=pixels_per_dim)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        # compute the function values on the grid
        f_s = self.create_f_s_pidm(X, Y, w, r) 
        self.f_s = f_s.unsqueeze(0).to(device)
        
    def create_f_s_pidm(self, x, y, w = 0.125, r = 10.):
        condition1 = torch.abs(x - 0.5 * w) <= 0.5 * w
        condition2 = torch.abs(x - 1 + 0.5 * w) <= 0.5 * w
        condition3 = torch.abs(y - 0.5 * w) <= 0.5 * w
        condition4 = torch.abs(y - 1 + 0.5 * w) <= 0.5 * w

        result = torch.zeros_like(x)
        result[torch.logical_and(condition1, condition3)] = r
        result[torch.logical_and(condition2, condition4)] = -r
        return result
    
    @classmethod
    def postprocess_darcy64(cls, x: torch.Tensor):
        assert x.shape[1] == 2 and x.ndim == 4
        a, u = x.chunk(2, dim=1)
        a = a * 23 + 23
        u = u * 1.7
        return torch.cat([a, u], dim=1)
    
    def forward(self, x0_pred: torch.Tensor, compute_bc: bool = False):
        assert x0_pred.shape[1] == 2 and x0_pred.ndim == 4
        if self.postprocess_input:
            x0_pred = self.postprocess_darcy64(x0_pred)
        permeability_field, p = x0_pred.chunk(2, dim=1)
        p_d0 = self.grads.stencil_gradients(p, mode='d_d0')
        p_d1 = self.grads.stencil_gradients(p, mode='d_d1')
        p_d00 = self.grads.stencil_gradients(p, mode='d_d00')
        p_d11 = self.grads.stencil_gradients(p, mode='d_d11')
        perm_d0 = self.grads.stencil_gradients(permeability_field, mode='d_d0')
        perm_d1 = self.grads.stencil_gradients(permeability_field, mode='d_d1')
        v00 = -permeability_field * p_d00 - perm_d0 * p_d0
        v11 = -permeability_field * p_d11 - perm_d1 * p_d1
        residual = v00 + v11 - self.f_s
        
        if compute_bc:
            grad_p = torch.cat([p_d0, p_d1], dim=1)
            residual_bc = torch.zeros_like(grad_p)
            residual_bc[:,0,0,:] = -grad_p[:,0,0,:] # xmin / top (acc. to matplotlib visualization)
            residual_bc[:,0,-1,:] = grad_p[:,0,-1,:] # xmax / bot
            residual_bc[:,1,:,0] = grad_p[:,1,:,0] # ymin / left
            residual_bc[:,1,:,-1] = -grad_p[:,1,:,-1] # ymax / right
            
            residual = torch.cat([residual, residual_bc], dim=1)
        return residual
    
class BurgersResidual(nn.Module):
    def __init__(self, postprocess_input: bool, domain_length=1.0, pixels_per_dim=128, device=torch.device('cuda')):
        super().__init__()
        self.postprocess_input = postprocess_input
        self.domain_length = domain_length
        self.pixels_per_dim = pixels_per_dim
        self.dx = domain_length / pixels_per_dim
        self.dt = domain_length / pixels_per_dim
        self.deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 1, 3) / (2 * self.dx)
        self.deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float32, device=device).view(1, 1, 3, 1) / (2 * self.dt)
        
    @classmethod
    def postprocess_burgers(cls, x: torch.Tensor):
        assert x.shape[1] == 1 and x.ndim == 4
        return x * 1.415
    
    def forward(self, u: torch.Tensor):
        if self.postprocess_input:
            u = self.postprocess_burgers(u)
        deriv_x = self.deriv_x.to(u)
        deriv_t = self.deriv_t.to(u)
        u_x = F.pad(u, pad=(1, 1, 0, 0), mode='replicate')
        u_x = F.conv2d(u_x, deriv_x)
        u_t = F.conv2d(F.pad(u, pad=(0, 0, 1, 1), mode='replicate'), deriv_t) 
        u_xx = F.conv2d(F.pad(u_x, pad=(1, 1, 0, 0), mode='replicate'), deriv_x)
        
        pde_loss = u_t + u * u_x - 0.01 * u_xx
        return pde_loss

class HelmholtzResidual(nn.Module):
    def __init__(self, postprocess_input: bool, domain_length=1.0, pixels_per_dim=128, k=1, device=torch.device('cuda')):
        super().__init__()
        self.postprocess_input = postprocess_input
        self.domain_length = domain_length
        self.pixels_per_dim = pixels_per_dim
        self.dx = domain_length / (pixels_per_dim - 1)
        self.k = k
        self.device = device
        
    @classmethod
    def postprocess_helmholtz(cls, x: torch.Tensor):
        assert x.shape[1] == 2 and x.ndim == 4
        a, u = x.chunk(2, dim=1)
        a = a * 2.15
        u = u * 0.028
        return torch.cat([a, u], dim=1)
    
    def forward(self, x: torch.Tensor):
        if self.postprocess_input:
            x = self.postprocess_helmholtz(x)
        a, u = x.chunk(2, dim=1)
        assert a.shape[2] == a.shape[3] == self.pixels_per_dim
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'replicate')
        d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
               u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / self.dx**2
        residual = d2u + self.k**2 * u - a
        return residual
    
class PoissonResidual(nn.Module):
    def __init__(self, postprocess_input: bool, domain_length=1.0, pixels_per_dim=128, device=torch.device('cuda')):
        super().__init__()
        self.postprocess_input = postprocess_input
        self.domain_length = domain_length
        self.pixels_per_dim = pixels_per_dim
        self.dx = domain_length / (pixels_per_dim - 1)
        self.device = device
    
    @classmethod
    def postprocess_poisson(cls, x: torch.Tensor):
        assert x.shape[1] == 2 and x.ndim == 4
        a, u = x.chunk(2, dim=1)
        a = a * 2.5
        u = u / 36.5
        return torch.cat([a, u], dim=1)
    
    def forward(self, x: torch.Tensor):
        if self.postprocess_input:
            x = self.postprocess_poisson(x)
        a, u = x.chunk(2, dim=1)
        assert a.shape[2] == a.shape[3] == self.pixels_per_dim
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'replicate')
        d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
               u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / self.dx**2
        residual = d2u - a
        return residual
        
        
class KolmogorovResidual(nn.Module):
    def __init__(self, postprocess_input: bool, re=1000.0, dt=1/32):
        super().__init__()
        self.postprocess_input = postprocess_input
        self.re = re
        self.dt = dt
        
    @classmethod
    def postprocess_kolmogorov(cls, x: torch.Tensor):
        assert x.shape[1] == 3 and x.ndim == 4
        x = x * 12
        return x
        
    def forward(self, w):
        if self.postprocess_input:
            w = self.postprocess_kolmogorov(w)
        nx = w.size(2)
        device = w.device
        w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
        # Wavenumbers in y-direction
        k_max = nx//2
        w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
        # Wavenumbers in y-direction
        k_max = nx//2
        N = nx
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
            reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
            reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
        # Negative Laplacian in Fourier space
        lap = (k_x ** 2 + k_y ** 2)
        lap[..., 0, 0] = 1.0
        psi_h = w_h / lap

        u_h = 1j * k_y * psi_h
        v_h = -1j * k_x * psi_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
        v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
        wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
        wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
        wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
        advection = u*wx + v*wy

        wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * self.dt)

        # establish forcing term
        x = torch.linspace(0, 2*torch.pi, nx + 1, device=device)
        x = x[0:-1]
        _, Y = torch.meshgrid(x, x, indexing='ij')
        f = -4*torch.cos(4*Y)

        residual = wt + (advection - (1.0 / self.re) * wlap + 0.1*w[:, 1:-1]) - f
        # residual_loss = (residual**2).mean()
        return residual
'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import torch
from torch.nn import functional as F
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, ifft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate A^T * Y + (I - A^T * A)X
        return self.transpose(measurement, **kwargs) + self.ortho_project(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        return data

    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return torch.zeros_like(data)

    def project(self, data, measurement, **kwargs):
        return measurement


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        
        # Adjoint operator for non-symmetric motion blur
        self.conv_transpose = Blurkernel(blur_type='motion',
                                         kernel_size=kernel_size,
                                         std=intensity,
                                         device=device).to(device)
        kernel_flipped = torch.flip(kernel, dims=[0, 1])
        self.conv_transpose.update_weights(kernel_flipped)
    
    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return self.conv_transpose(data)

    def get_kernel(self):
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        # Gaussian kernel is symmetric
        return self.forward(data, **kwargs)

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', None)
        if mask is None:
            raise ValueError("InpaintingOperator requires a 'mask' in kwargs.")
        return data * mask.to(self.device)
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

    def project(self, data, measurement, **kwargs):
        return measurement + self.ortho_project(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        # fft2_m is used for multi-channel/multi-coil if needed
        # but here we assume single channel or independent channels
        amplitude = fft2_m(padded).abs()
        return amplitude

    def project(self, data, measurement, **kwargs):
        # Better projection for Phase Retrieval:
        # Keep phase of current estimate, replace amplitude with measurement
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        freq = fft2_m(padded)
        phase = torch.angle(freq)
        
        # Reconstruct in Fourier space
        reconstructed_freq = measurement * torch.exp(1j * phase)
        
        # Back to image space
        reconstructed_padded = ifft2_m(reconstructed_freq).real
        
        # Crop back to original size
        if self.pad > 0:
            return reconstructed_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return reconstructed_padded

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        # Persistent kernel for consistent forward operator
        self.kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        # Use stored kernel unless one is provided in kwargs
        kernel = kwargs.get('kernel', self.kernel)
        data_norm = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data_norm, kernel=kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        """
        Follow skimage.util.random_noise.
        Optimized to use torch.poisson to stay on the same device.
        """
        # Scale to [0, 1] range for poisson sampling
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        
        # Scale by rate and 255 (standard for 8-bit image noise simulation)
        rate = data * 255.0 * self.rate
        
        # Sample from Poisson distribution
        # torch.poisson expects the rate (lambda) as input
        noisy_data = torch.poisson(rate) / 255.0 / self.rate
        
        # Scale back to [-1, 1]
        noisy_data = noisy_data * 2.0 - 1.0
        noisy_data = noisy_data.clamp(-1, 1)
        
        return noisy_data

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
from abc import ABC, abstractmethod
import torch
import math

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='rl_ps')
class RLPosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, rl_eta=None, **kwargs):
        # We expect the diffusion loop to pass 'rl_eta' from the policy network
        if rl_eta is None:
            raise ValueError("rl_eta must be provided by the RL policy network.")

        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            **kwargs
        )

        # Reshape rl_eta for broadcasting across the batch/channels
        while rl_eta.ndim < norm_grad.ndim:
            rl_eta = rl_eta.unsqueeze(-1)

        # Apply the dynamically learned step size!
        x_t = x_t - rl_eta * norm_grad

        return x_t, norm

@register_conditioning_method(name='adaptive_ps')
class AdaptivePosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.eta_min = kwargs.get('eta_min', 0.05)
        self.eta_max = kwargs.get('eta_max', 0.35)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)

    def get_adaptive_eta(self, t):
        """
        t: current reverse timestep tensor, shape [B]
        We define progress:
            progress = 0 at beginning of reverse sampling
            progress = 1 at final stage
        """
        progress = 1.0 - t.float() / float(self.num_timesteps - 1)

        # small -> large -> small
        eta = self.eta_min + (self.eta_max - self.eta_min) * torch.sin(math.pi * progress)

        return eta

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t=None, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            **kwargs
        )

        if t is None:
            raise ValueError("AdaptivePosteriorSampling requires timestep t.")

        eta = self.get_adaptive_eta(t)

        # reshape eta for broadcasting
        while eta.ndim < norm_grad.ndim:
            eta = eta.unsqueeze(-1)

        x_t = x_t - eta * norm_grad

        return x_t, norm
    
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

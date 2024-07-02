import numpy as np 
import jax
import jax.numpy as jnp
from functools import partial

# def onset_1d(target):
#     stft = jax.scipy.signal.stft(target,boundary='even') # create spectrogram 
#     norm_spec = jnp.abs(stft[2])[0]**0.5 # normalize the spectrogram
#     kernel = gaussian_kernel1d(3,0,10) #create a gaussian kernel (sigma,order,radius)
#     ts = norm_spec.sum(axis=0) # calculate amplitude changes 
#     onsets = jnp.convolve(ts,kernel,mode="same") # smooth amplitude curve 
#     return onsets

@partial(jax.jit, static_argnames=["sf"])
def onset_1d(target,k,sf):
    # print(target.shape)
    ts = sf(target)[0].sum(axis=1)
    onsets = jnp.convolve(ts, k, mode="same")  # smooth amplitude curve
    return onsets

def hell():
    print("hello")

def gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    copied directly from: https://github.com/scipy/scipy/blob/v1.14.0/scipy/ndimage/_filters.py#L186C1-L215C1
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x

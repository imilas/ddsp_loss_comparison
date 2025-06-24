import numpy as np 
import jax
import jax.numpy as jnp
from functools import partial
from audax.core import functional

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

def spec_func(nfft,win_len,hop_len):
    # creates a spectrogram helper
    window = jnp.hanning(win_len)
    spec_func = partial(
        functional.spectrogram,
        pad=0,
        window=window,
        n_fft=nfft,
        hop_length=hop_len,
        win_length=win_len,
        power=1,
        normalized=True,
        center=True,
        onesided=True,
    )

    return spec_func

def return_mel_spec(NFFT,SR=44100):
    WIN_LEN = 400
    HOP_LEN = 20
    window = jnp.hanning(WIN_LEN)
    spec_func = partial(functional.spectrogram, pad=0, window=window, n_fft=NFFT,
                       hop_length=HOP_LEN, win_length=WIN_LEN, power=1,
                       normalized=False, center=False, onesided=True)
    fb = functional.melscale_fbanks(n_freqs=(NFFT//2)+1, n_mels=32,
                             sample_rate=SR, f_min=60., f_max=SR//2)
    mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)

    jax_spec = jax.jit(spec_func)
    mel_spec = jax.jit(mel_spec_func) 
    return mel_spec,jax_spec

def norm_sound(sound):
    return sound / jnp.max(sound)

def single_level_loss(mel_fun, spec_fun, p, t):
    p_spec = mel_fun(spec_fun(p))
    t_spec = mel_fun(spec_fun(t))
    loss = jnp.abs(p_spec - t_spec) + jnp.abs(jnp.log(p_spec) - jnp.log(t_spec))
    return loss, p_spec, t_spec

def loss_multi_spec(prediction, target,spec_funs):
    # Normalize sound signals
    prediction = norm_sound(prediction)
    target = norm_sound(target)
    
    # Calculate losses at each level
    loss_0, _, _ = single_level_loss(spec_funs[0][0], spec_funs[0][1], prediction, target)
    loss_1, _, _ = single_level_loss(spec_funs[1][0], spec_funs[1][1], prediction, target)
    loss_2, _, _ = single_level_loss(spec_funs[2][0], spec_funs[2][1], prediction, target)
    loss_3, _, _ = single_level_loss(spec_funs[3][0], spec_funs[3][1], prediction, target)
    
    # Aggregate losses across all levels and take the mean
    loss = jnp.mean(loss_0) + jnp.mean(loss_1) + jnp.mean(loss_2) + jnp.mean(loss_3)
    return jnp.mean(loss)

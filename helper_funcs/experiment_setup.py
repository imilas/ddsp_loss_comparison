# helper_funcs/experiment_setup.py

import jax.numpy as jnp
import numpy as np
import random
from kymatio.jax import Scattering1D
from helper_funcs import loss_helpers, softdtw_jax

SAMPLE_RATE = 44100
spec_funs = [loss_helpers.return_mel_spec(x, SAMPLE_RATE) for x in [512, 1024, 2048, 4096]]
MSS_loss = loss_helpers.loss_multi_spec

# Basic losses
naive_loss = lambda x, y: jnp.abs(x - y).mean()
cosine_distance = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Spectrogram
NFFT = 512
WIN_LEN = 600
HOP_LEN = 100
spec_func = loss_helpers.spec_func(NFFT, WIN_LEN, HOP_LEN)

def clip_spec(x):
    return jnp.clip(x, a_min=0, a_max=1)

# SoftDTW and kernel
dtw_jax = softdtw_jax.SoftDTW(gamma=1)
kernel = jnp.array(loss_helpers.gaussian_kernel1d(3, 0, 10))

# Scattering
J = 12
Q = 2
scat_jax = Scattering1D(J, SAMPLE_RATE, Q)

# Onset function
onset_1d = loss_helpers.onset_1d

# Multi-spectrograms
spec_funs = [loss_helpers.return_mel_spec(x, SAMPLE_RATE) for x in [512, 1024, 2048, 4096]]

import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import sys
    sys.path.append("../")

    from functools import partial
    import itertools
    from pathlib import Path
    import os
    import jax
    import jax.numpy as jnp
    from jax import random

    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state
    from flax.core.frozen_dict import unfreeze
    import optax

    from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
    import dawdreamer.faust.box as fbox 

    from tqdm.notebook import tqdm
    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc

    from helpers import faust_to_jax as fj
    from helpers import loss_helpers
    from helpers import softdtw_jax
    from kymatio.jax import Scattering1D

    from IPython.display import HTML
    from IPython.display import Audio
    import IPython.display as ipd
    return (
        Audio,
        FaustContext,
        HTML,
        Path,
        Scattering1D,
        animation,
        createLibContext,
        destroyLibContext,
        fbox,
        fj,
        ipd,
        itertools,
        jax,
        jnp,
        librosa,
        loss_helpers,
        mo,
        nn,
        np,
        optax,
        os,
        partial,
        plt,
        random,
        rc,
        softdtw_jax,
        sys,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(Audio, ipd):
    SAMPLE_RATE = 44100

    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.:
            data /= abs(data).max()
        ipd.display(Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay))
    return SAMPLE_RATE, show_audio


@app.cell
def __(SAMPLE_RATE, Scattering1D, jnp, loss_helpers, np, softdtw_jax):
    # distance functions
    naive_loss = lambda x, y: jnp.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # Spec function
    NFFT = 512
    WIN_LEN = 600
    HOP_LEN = 100
    spec_func = loss_helpers.spec_func(NFFT, WIN_LEN, HOP_LEN)


    def clip_spec(x):
        return jnp.clip(x, a_min=0, a_max=1)


    dtw_jax = softdtw_jax.SoftDTW(gamma=1)

    kernel = jnp.array(loss_helpers.gaussian_kernel1d(3, 0, 10))  # create a gaussian kernel (sigma,order,radius)

    J = 6  # higher creates smoother loss but more costly
    Q = 1
    scat_jax = Scattering1D(J, SAMPLE_RATE, Q)

    onset_1d = loss_helpers.onset_1d
    return (
        HOP_LEN,
        J,
        NFFT,
        Q,
        WIN_LEN,
        clip_spec,
        cosine_distance,
        dtw_jax,
        kernel,
        naive_loss,
        onset_1d,
        scat_jax,
        spec_func,
    )


@app.cell
def __(
    SAMPLE_RATE,
    fj,
    instrument_jit,
    instrument_params,
    mo,
    noise,
    true_instrument_params,
    true_noise,
):
    mo.output.clear()
    init_sound = instrument_jit(instrument_params, noise, SAMPLE_RATE)[0]
    target_sound = instrument_jit(true_instrument_params, true_noise, SAMPLE_RATE)[0]
    fj.show_audio(init_sound)
    fj.show_audio(target_sound)
    return init_sound, target_sound


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

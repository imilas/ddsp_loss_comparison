import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import functools
    from functools import partial
    import itertools
    from pathlib import Path
    import os
    import jax
    import jax.numpy as jnp

    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state
    from flax.core.frozen_dict import unfreeze
    import optax


    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt

    from helpers import faust_to_jax as fj
    from audax.core import functional
    import copy
    from helpers import ts_comparisions as ts_comparisons
    import dtw

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be


    naive_loss = lambda x, y: np.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )
    return (
        Path,
        SAMPLE_RATE,
        copy,
        cosine_distance,
        default_device,
        dtw,
        fj,
        functional,
        functools,
        itertools,
        jax,
        jnp,
        length_seconds,
        librosa,
        mo,
        naive_loss,
        nn,
        np,
        optax,
        os,
        partial,
        plt,
        train_state,
        ts_comparisons,
        unfreeze,
        wavfile,
    )


@app.cell
def __(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code_target = """
    import("stdfaust.lib");
    f = hslider("freq",1.,0,5,0.1);
    peak_f = hslider("peak_f",40,40,200,1);
    process = os.saw4(os.osc(f)*peak_f);
    """

    faust_code_instrument = """
    import("stdfaust.lib");
    f = hslider("freq",2.,0,5,0.1);
    peak_f = hslider("peak_f",60,40,200,1);
    process = os.saw4(os.osc(f)*peak_f);
    """
    return faust_code_instrument, faust_code_target, key


@app.cell
def __(
    SAMPLE_RATE,
    faust_code_instrument,
    fj,
    jax,
    key,
    length_seconds,
    partial,
):
    instrument = fj.faust2jax(faust_code_instrument)
    instrument = instrument(SAMPLE_RATE)
    instrument_jit = jax.jit(
        partial(instrument.apply, mutable="intermediates"), static_argnums=[2]
    )
    noise = jax.random.uniform(
        jax.random.PRNGKey(10),
        [instrument.getNumInputs(), SAMPLE_RATE * length_seconds],
        minval=-1,
        maxval=1,
    )
    instrument_params = instrument.init(key, noise, SAMPLE_RATE)
    return instrument, instrument_jit, instrument_params, noise


@app.cell
def __(
    SAMPLE_RATE,
    faust_code_target,
    fj,
    instrument_jit,
    instrument_params,
    jax,
    mo,
    noise,
):
    mo.output.clear()
    init_sound = instrument_jit(instrument_params, noise, SAMPLE_RATE)[0]
    target_sound = fj.process_noise_in_faust(
        faust_code_target, jax.random.PRNGKey(10)
    )[0]
    fj.show_audio(target_sound)
    fj.show_audio(init_sound)
    return init_sound, target_sound


@app.cell
def __(mo):
    mo.md(
        """
    Let's setup an SGD experiment with a customizable loss function. 
    """
    )
    return


@app.cell
def __(
    SAMPLE_RATE,
    instrument,
    instrument_params,
    jax,
    jnp,
    noise,
    optax,
    target_sound,
    train_state,
):
    learning_rate = 2e-4
    momentum = 0.9
    # Create Train state
    tx = optax.sgd(learning_rate, momentum)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )


    @jax.jit
    def train_step(state, x, y):
        """Train for a single step."""

        def loss_fn(params):
            pred = instrument.apply(params, x, SAMPLE_RATE)
            # L1 time-domain loss
            loss = (jnp.abs(pred - y)).mean()
            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss


    train_step(state, noise, target_sound)
    return learning_rate, momentum, state, train_step, tx


@app.cell
def __(target_sound):
    target_sound
    return


@app.cell
def __(state):
    state.params["params"]
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

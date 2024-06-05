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
    from jax import random

    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state
    from flax.core.frozen_dict import unfreeze
    import optax

    from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
    import dawdreamer.faust.box as fbox

    from tqdm.notebook import tqdm
    import time
    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc

    from IPython.display import HTML
    from IPython.display import Audio
    import IPython.display as ipd

    from kymatio.jax import Scattering1D

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)
    return (
        Audio,
        FaustContext,
        HTML,
        Path,
        Scattering1D,
        animation,
        createLibContext,
        default_device,
        destroyLibContext,
        fbox,
        functools,
        ipd,
        itertools,
        jax,
        jnp,
        librosa,
        mo,
        nn,
        np,
        optax,
        os,
        partial,
        plt,
        random,
        rc,
        time,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(Audio, ipd):
    SAMPLE_RATE = 44100


    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.0:
            data /= abs(data).max()
        ipd.display(
            Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)
        )
    return SAMPLE_RATE, show_audio


@app.cell
def __(FaustContext, SAMPLE_RATE, fbox, jnp):
    def make_sine(freq: float, T: int, sr=SAMPLE_RATE):
        """Return sine wave based on freq in Hz and duration T in samples"""
        return jnp.sin(jnp.pi * 2.0 * freq * jnp.arange(T) / sr)


    faust_code = f"""
    import("stdfaust.lib");
    cutoff = hslider("cutoff", 440., 20., 20000., .01);
    FX = fi.lowpass(1, cutoff);
    replace = !,_;
    process = ["cutoff":replace -> FX];
    """

    module_name = "MyDSP"

    with FaustContext():

        box = fbox.boxFromDSP(faust_code)

        jax_code = fbox.boxToSource(
            box, "jax", module_name, ["-a", "jax/minimal.py"]
        )

    custom_globals = {}
    exec(jax_code, custom_globals)
    FilterModel = custom_globals[module_name]
    return (
        FilterModel,
        box,
        custom_globals,
        faust_code,
        jax_code,
        make_sine,
        module_name,
    )


@app.cell
def __(FilterModel, SAMPLE_RATE, jax, jnp, make_sine, nn, np, partial):
    # We don't vmap FilterModel in-place because we need the original version inside AutomationModel
    HiddenModel = nn.vmap(
        FilterModel,
        in_axes=(0, None),
        variable_axes={"params": None},
        split_rngs={"params": False},
    )

    hidden_model = HiddenModel(SAMPLE_RATE)


    class AutomationModel_(nn.Module):

        automation_samples: int

        def getNumInputs(self):
            return 1

        @nn.compact
        def __call__(self, x, T: int) -> jnp.array:
            # make the learnable cutoff automation parameter.
            freq = self.param("freq", nn.initializers.constant(3), 1)
            angles = jnp.linspace(0, 1, T, endpoint=False)
            automation = jnp.sin(freq * angles * 2 * np.pi)
            automation = jnp.expand_dims(automation, axis=0)
            automation = jnp.interp(
                automation, jnp.array([-1, 1]), jnp.array([20, 20000])
            )
            self.sow("intermediates", "freq", freq)
            self.sow("intermediates", "cutoff", automation)

            x = jnp.concatenate([automation, x], axis=-2)
            filterModel = FilterModel(sample_rate=SAMPLE_RATE)
            audio = filterModel(x, T)

            return audio


    # set control rate to 100th of audio rate
    AUTOMATION_DOWNSAMPLE = 100  # @param {type:"integer"}
    RECORD_DURATION = 1.0  # @param {type:"number"}
    T = int(RECORD_DURATION * SAMPLE_RATE)
    automation_samples = T // AUTOMATION_DOWNSAMPLE

    jit_inference_fn = jax.jit(
        partial(
            AutomationModel_(automation_samples=automation_samples).apply,
            mutable="intermediates",
        ),
        static_argnums=[2],
    )
    AutomationModel = nn.vmap(
        AutomationModel_,
        in_axes=(0, None),
        variable_axes={"params": None},
        split_rngs={"params": False},
    )

    train_model = AutomationModel(automation_samples=automation_samples)

    batch_size = 1  # @param {type: 'integer'}
    hidden_automation_freq = 10.0  # @param {type:"number"}
    hidden_automation = 10_000 + make_sine(hidden_automation_freq, T) * 9500
    jnp.expand_dims(hidden_automation, axis=0)
    hidden_automation = jnp.tile(hidden_automation, (batch_size, 1, 1))
    print("hidden_automation shape: ", hidden_automation.shape)
    return (
        AUTOMATION_DOWNSAMPLE,
        AutomationModel,
        AutomationModel_,
        HiddenModel,
        RECORD_DURATION,
        T,
        automation_samples,
        batch_size,
        hidden_automation,
        hidden_automation_freq,
        hidden_model,
        jit_inference_fn,
        train_model,
    )


@app.cell
def __(
    T,
    batch_size,
    hidden_automation,
    hidden_model,
    jnp,
    random,
    train_model,
):
    key = random.PRNGKey(420)
    input_shape = (batch_size, train_model.getNumInputs(), T)

    key, subkey = random.split(key)
    hidden_inputs = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

    hidden_inputs = jnp.concatenate([hidden_automation, hidden_inputs], axis=1)
    print("hidden_shape:", hidden_inputs.shape)
    key, subkey = random.split(key)
    hidden_params = hidden_model.init({"params": subkey}, hidden_inputs, T)
    print("hidden params", hidden_params)
    key, subkey = random.split(key)
    rand_x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

    train_inputs = rand_x
    print("train_inputs shape:", train_inputs.shape)

    key, subkey = random.split(key)
    train_params = train_model.init({"params": subkey}, train_inputs, T)
    # print("train params:", train_params["params"].keys())
    print(" train params shape:", train_params["params"])
    return (
        hidden_inputs,
        hidden_params,
        input_shape,
        key,
        rand_x,
        subkey,
        train_inputs,
        train_params,
    )


@app.cell
def __(
    T,
    hidden_inputs,
    hidden_model,
    show_audio,
    train_inputs,
    train_model,
    train_params,
):
    hidden_sounds = hidden_model.apply({}, hidden_inputs, T)
    train_sounds = train_model.apply(train_params, train_inputs, T)
    show_audio(hidden_sounds[0])
    show_audio(train_sounds[0])
    return hidden_sounds, train_sounds


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

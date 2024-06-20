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
def __(mo):
    mo.md(
        """
    ```FilterModel```: Takes in a series of values which control the cutoff parameter. These values (or automation) is the first parameter, and audio is the second. 

    ```HiddenModel```: Is a vmapped FilterModel

    ```AutomationModel```: Has parameter 'freq' which determines the cutoff oscillation frequency. It takes in x/audio and T as arguments. Is vmapped in placed.

    ```jit_inference_fn```: Partially jitted AutomationModel for inference. Is not vmapped? 

    ``` train_model```: instance of Automation model, with static argument set. 

    ``` hidden_automation_*```: true values we are approximating 

    """
    )
    return


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
def __(T, jnp, partial, train_sounds):
    from audax.core import functional

    NFFT = 256
    WIN_LEN = 400
    HOP_LEN = 160

    # creates a spectrogram helper
    window = jnp.hanning(WIN_LEN)
    spec_func = partial(
        functional.spectrogram,
        pad=0,
        window=window,
        n_fft=NFFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        power=2.0,
        normalized=False,
        center=True,
        onesided=True,
    )
    fb = functional.melscale_fbanks(
        n_freqs=(NFFT // 2) + 1, n_mels=64, sample_rate=T, f_min=60.0, f_max=7800.0
    )
    mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)
    jax_spec = spec_func(train_sounds[0][0])
    mel_spec = mel_spec_func(jax_spec)  # output of shape (1, 101, 64)
    return (
        HOP_LEN,
        NFFT,
        WIN_LEN,
        fb,
        functional,
        jax_spec,
        mel_spec,
        mel_spec_func,
        spec_func,
        window,
    )


@app.cell
def __(jax_spec, plt):
    plt.imshow(
        jax_spec[0].T,
        aspect=2,
        cmap=plt.cm.gray_r,
        origin="lower",
    )
    # plt.imshow(
    #     spec_func(train_sounds[0])[0].T,
    #     aspect=2,
    #     cmap=plt.cm.gray_r,
    #     origin="lower",
    # )
    plt.ylabel("FFT Bin Number")
    plt.xlabel("Frame Number")
    plt.title("Spectrograms")
    return


@app.cell
def __(mo):
    mo.md(
        """here we use the fourier representation of audio to calculate the loss. The loss function is $\sum |d_{i,j}|$ where $d = \chi_x - \chi_y$, $\chi$ is the spectrogram representation, and $x$ is the output of the synthesizer, and $y$ is the sound we're approximating. The dimensions $i$ and $j$ depend on the length of the sound and the parameters of the FFT function.

    The issue with this representation is that d is sensetive to the alignment of sounds. If x is a copy of y but offset by a few time-steps, the loss function would return a large value. 

    An improvement to this setup could be to convolve the loss functions and return the largest value as a similarity metric. 
    """
    )
    return


@app.cell
def __(
    T,
    hidden_automation,
    hidden_model,
    hidden_params,
    input_shape,
    jax,
    jit_inference_fn,
    jnp,
    key,
    np,
    optax,
    random,
    spec_func,
    tqdm,
    train_model,
    train_params,
    train_state,
):
    learning_rate = 2  # @param {type: 'number'}
    momentum = 0.95  # @param {type: 'number'}

    # Create Train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=train_model.apply, params=train_params, tx=tx
    )


    @jax.jit
    def train_step(state, x, y):
        """Train for a single step."""

        def loss_fn(params):
            pred = train_model.apply(params, x, T)

            ## spec loss
            pred_t = spec_func(pred[0][0])
            y_t = spec_func(y[0][0])
            loss = (jnp.abs(pred_t - y_t)).mean()

            # # L1 time-domain loss
            # loss = (jnp.abs(pred - y)).mean()

            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss


    losses = []
    cutoffs = []

    train_steps = 200
    steps_per_eval = 1

    pbar = tqdm(range(train_steps))

    jit_hidden_model = jax.jit(hidden_model.apply, static_argnums=[2])

    # Data containers for animation
    cutoff_data = []
    freq_data = []
    for n in pbar:
        print(n, end="\r")
        key1, subkey1 = random.split(key)
        # generate random signal to be filtered
        x = random.uniform(subkey1, shape=input_shape, minval=-1, maxval=1)

        # concat hidden automation to random signal
        automation_and_x = jnp.concatenate([hidden_automation, x], axis=1)

        # get ground truth of filtered signal
        y = jit_hidden_model(hidden_params, automation_and_x, T)

        # optimize learned automation based on input signal and GT filtered signal
        state, loss = train_step(state, x, y)

        if n % steps_per_eval == 0:
            audio, mod_vars = jit_inference_fn(state.params, x[0], T)
            cutoff = np.array(mod_vars["intermediates"]["cutoff"])[0]
            cutoff_data.append(cutoff)
            freq_data.append(mod_vars["intermediates"]["freq"][0])
            pbar.set_description(f"loss: {loss:.2f}")
            losses.append(loss)
    return (
        audio,
        automation_and_x,
        cutoff,
        cutoff_data,
        cutoffs,
        freq_data,
        jit_hidden_model,
        key1,
        learning_rate,
        loss,
        losses,
        mod_vars,
        momentum,
        n,
        pbar,
        state,
        steps_per_eval,
        subkey1,
        train_step,
        train_steps,
        tx,
        x,
        y,
    )


@app.cell
def __(freq_data, losses, plt):
    fig_1, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("loss", color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "OSC-frequency", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(freq_data, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    # ax2.set_yscale("log")
    fig_1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return ax1, ax2, color, fig_1


@app.cell
def __(T, jnp, spec_func, train_model, x, y):
    # loss landscape
    def loss_fn_sigdiff(params):
        pred = train_model.apply(params, x, T)

        ## spec loss
        pred_t = spec_func(pred[0][0])
        y_t = spec_func(y[0][0])
        loss = (jnp.abs(pred_t - y_t)).mean()

        # # L1 time-domain loss
        # loss = (jnp.abs(pred - y)).mean()
        return loss
    return loss_fn_sigdiff,


@app.cell
def __(freq_data, jax, jnp, loss_fn_sigdiff, np):
    def test_freq(f):
        return loss_fn_sigdiff({"params": {"freq": jnp.array([f])}})


    test_freqs = jax.vmap(test_freq, 0)
    test_freqs_jitted = jax.jit(test_freqs)
    freqs_explored = [np.min(freq_data) - 40, np.max([np.max(freq_data) + 1, 13])]
    oscillation_frequencies = jnp.linspace(
        freqs_explored[0], freqs_explored[1], 1000
    )
    test_losses = test_freqs_jitted(oscillation_frequencies)
    return (
        freqs_explored,
        oscillation_frequencies,
        test_freq,
        test_freqs,
        test_freqs_jitted,
        test_losses,
    )


if __name__ == "__main__":
    app.run()

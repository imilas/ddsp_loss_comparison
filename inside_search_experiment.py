import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        f"""
    This is a streamlined experiment for testing out loss functions. We manually define a synthesizer program containing _target parameters_ which make a _target sound_\. <br>
    In the experiments, the target paramaters are changed from their original values. The goal of the search function is to find the target parameters by comparing the output of the synthesizer to the target sound. This comparison is done via the _loss function_\. We analyize the effect of the loss functions on the search process. <br>
    The main variable of interest is whether or not the process successfully finds the target parameter, but we also care about speed and steps it takes to find the target.
    """
    )
    return


@app.cell
def __():
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
    from audax.core import functional
    import copy
    import dm_pix

    from helpers import faust_to_jax as fj
    from helpers import onsets
    from helpers import softdtw_jax
    from kymatio.jax import Scattering1D


    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return (
        Path,
        SAMPLE_RATE,
        Scattering1D,
        copy,
        default_device,
        dm_pix,
        fj,
        functional,
        functools,
        itertools,
        jax,
        jnp,
        length_seconds,
        librosa,
        nn,
        np,
        onsets,
        optax,
        os,
        partial,
        plt,
        softdtw_jax,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    true_params = {"lp_cut": 2000}
    init_params = {"lp_cut": 20000}

    faust_code_target = f"""
    import("stdfaust.lib");
    cutoff = hslider("lp_cut",{true_params["lp_cut"]},101,20000,1);
    FX = fi.lowpass(5,cutoff);
    process = os.osc(os.osc(3))*1000,_:["lp_cut":+(_,_)->FX];
    """

    faust_code_instrument = f"""
    import("stdfaust.lib");
    cutoff = hslider("lp_cut",{init_params["lp_cut"]},101,20000,1);
    FX = fi.lowpass(5,cutoff);
    process = os.osc(os.osc(3))*1000,_:["lp_cut":+(_,_)->FX];
    """
    return (
        faust_code_instrument,
        faust_code_target,
        init_params,
        key,
        true_params,
    )


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
    print(instrument_params)
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
        """Let's define the losses:
    """
    )
    return


@app.cell
def __(functional, jnp, np, partial):
    # distance functions
    naive_loss = lambda x, y: jnp.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    # Spec function
    NFFT = 512
    WIN_LEN = 800
    HOP_LEN = 200
    # creates a spectrogram helper
    window = jnp.hanning(WIN_LEN)
    spec_func = partial(
        functional.spectrogram,
        pad=0,
        window=window,
        n_fft=NFFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        power=1,
        normalized=True,
        center=True,
        onesided=True,
    )


    def clipped_spec(x):
        jax_spec = spec_func(x)
        jax_spec = jnp.clip(jax_spec, a_min=0, a_max=1)
        return jax_spec
    return (
        HOP_LEN,
        NFFT,
        WIN_LEN,
        clipped_spec,
        cosine_distance,
        naive_loss,
        spec_func,
        window,
    )


@app.cell
def __(softdtw_jax):
    dtw_jax = softdtw_jax.SoftDTW(gamma=0.8)
    return dtw_jax,


@app.cell
def __(SAMPLE_RATE, Scattering1D, jnp, onsets):
    kernel = jnp.array(
        onsets.gaussian_kernel1d(3, 0, 10)
    )  # create a gaussian kernel (sigma,order,radius)

    J = 6  # higher creates smoother loss but more costly
    Q = 1

    scat_jax = Scattering1D(J, SAMPLE_RATE, Q)
    return J, Q, kernel, scat_jax


@app.cell
def __(onsets):
    onset_1d = onsets.onset_1d
    return onset_1d,


@app.cell
def __(
    SAMPLE_RATE,
    init_params,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    naive_loss,
    noise,
    np,
    optax,
    scat_jax,
    target_sound,
    train_state,
    true_params,
):
    learning_rate = 0.05
    # Create Train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )


    # loss fn shows the difference between the output of synth and a target_sound
    def loss_fn(params):
        pred = instrument.apply(params, noise, SAMPLE_RATE)
        # L1 time-domain loss
        loss = 0
        # loss = (jnp.abs(pred - target_sound)).mean()
        # loss = naive_loss(spec_func(pred)[0],target_spec)
        # loss = 1/dm_pix.ssim(clipped_spec(target_sound),clipped_spec(pred))
        # loss = dm_pix.simse(clipped_spec(target_sound),clipped_spec(pred))
        # loss = dtw_jax(pred,target_sound)
        # loss = dtw_jax(onset_1d(target_sound,kernel,spec_func), onset_1d(pred,kernel,spec_func))
        loss = naive_loss(scat_jax(target_sound),scat_jax(pred))
        # jax.debug.print("onsets:{y},loss:{l}",y=onsets.onset_1d(pred)[0:3],l=loss)
        return loss, pred


    @jax.jit
    def train_step(state):
        """Train for a single step."""
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss


    losses = []
    sounds = []
    search_params = {
        k: [init_params[k]] for k in true_params.keys()
    }  # will record parameters while searching
    for n in range(200):
        state, loss = train_step(state)
        if n % 1 == 0:
            audio, mod_vars = instrument_jit(state.params, noise, SAMPLE_RATE)
            sounds.append(audio)

            for pname in search_params.keys():
                parameter_value = np.array(
                    mod_vars["intermediates"]["dawdreamer/%s" % pname]
                )[0]
                search_params[pname].append(parameter_value)
            losses.append(loss)
            # print(n, loss, state.params)
            print(n,end="\r")
    return (
        audio,
        learning_rate,
        loss,
        loss_fn,
        losses,
        mod_vars,
        n,
        parameter_value,
        pname,
        search_params,
        sounds,
        state,
        train_step,
        tx,
    )


@app.cell
def __(losses, mo, plt, search_params, true_params):
    mo.output.clear()
    fig_1, ax1 = plt.subplots()
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("loss", color="black")
    ax1.plot(losses, color="black")
    # ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    c = 0
    colors = ["red", "green", "blue", "purple"]
    for pname2, pvalue in search_params.items():
        ax2.set_ylabel(
            pname2, color=colors[c]
        )  # we already handled the x-label with ax1
        ax2.plot(search_params[pname2], color=colors[c])
        plt.axhline(
            true_params[pname2], linestyle="dashed", color=colors[c], label="true"
        )
        ax2.tick_params(axis="y")

        c += 1
        # color = "tab:green"
        # ax2.plot(search_params["lp_cut"], color=color)
        # ax2.tick_params(axis="y", labelcolor=color)

    fig_1.tight_layout()  # otherwise the right y-label is slightly clipped


    plt.show()
    return ax1, ax2, c, colors, fig_1, pname2, pvalue


@app.cell
def __():
    import helpers.dfta_core as dfta_core

    return dfta_core,


@app.cell
def __(dfta_core, fj, mo):
    mo.output.clear()
    import torch
    theta = [torch.tensor(200), torch.tensor(1.), torch.tensor(1.8)]
    signal = dfta_core.generate_am_chirp(theta, bw=5, duration=1, sr=44100, delta=100)
    fj.show_audio(signal)
    return signal, theta, torch


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

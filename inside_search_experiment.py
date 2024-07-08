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

    # from audax.core import functional
    import copy
    import dm_pix

    from helpers import faust_to_jax as fj
    from helpers import loss_helpers
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
        functools,
        itertools,
        jax,
        jnp,
        length_seconds,
        librosa,
        loss_helpers,
        nn,
        np,
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
def __(mo):
    mo.md(
        """Let's define the losses
    """
    )
    return


@app.cell
def __(SAMPLE_RATE, Scattering1D, jnp, loss_helpers, np, softdtw_jax):
    # distance functions
    naive_loss = lambda x, y: jnp.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    # Spec function
    NFFT = 512
    WIN_LEN = 800
    HOP_LEN = 200
    spec_func = loss_helpers.spec_func(NFFT, WIN_LEN, HOP_LEN)


    def clip_spec(x):
        return jnp.clip(x, a_min=0, a_max=1)


    dtw_jax = softdtw_jax.SoftDTW(gamma=0.8)

    kernel = jnp.array(
        loss_helpers.gaussian_kernel1d(3, 0, 10)
    )  # create a gaussian kernel (sigma,order,radius)

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
def __(SAMPLE_RATE, fj, jax, length_seconds, mo, partial):
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
    mo.output.clear()
    init_sound = instrument_jit(instrument_params, noise, SAMPLE_RATE)[0]
    target_sound = fj.process_noise_in_faust(
        faust_code_target, jax.random.PRNGKey(10)
    )[0]
    return (
        faust_code_instrument,
        faust_code_target,
        init_params,
        init_sound,
        instrument,
        instrument_jit,
        instrument_params,
        key,
        noise,
        target_sound,
        true_params,
    )


@app.cell
def __(fj, init_sound, target_sound):
    fj.show_audio(target_sound)
    fj.show_audio(init_sound)
    return


@app.cell
def __(
    SAMPLE_RATE,
    clip_spec,
    dm_pix,
    init_params,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    noise,
    np,
    optax,
    spec_func,
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
        loss = dm_pix.simse(
            clip_spec(spec_func(target_sound)), clip_spec(spec_func(pred))
        )
        # loss = dtw_jax(pred,target_sound)
        # loss = dtw_jax(onset_1d(target_sound,kernel,spec_func), onset_1d(pred,kernel,spec_func))
        # loss = naive_loss(scat_jax(target_sound),scat_jax(pred))
        # jax.debug.print("loss_helpers:{y},loss:{l}",y=loss_helpers.onset_1d(pred)[0:3],l=loss)
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
            print(n, end="\r")
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
def __(mo):
    mo.md(
        """quiver plots
    1. Define grad function using loss
    2. draw quivers
    """
    )
    return


@app.cell
def __(
    SAMPLE_RATE,
    clip_spec,
    dm_pix,
    instrument,
    jax,
    noise,
    spec_func,
    state,
    target_sound,
):
    # loss fn shows the difference between the output of synth and a target_sound
    def loss_fn2(params):
        pred = instrument.apply(params, noise, SAMPLE_RATE)
        # loss = (jnp.abs(pred - target_sound)).mean()
        # loss = naive_loss(spec_func(pred)[0],spec_func(target_sound)[0])

        loss = dm_pix.simse(
            clip_spec(spec_func(target_sound)), clip_spec(spec_func(pred))
        )
        return loss, pred


    grad_fn = jax.value_and_grad(loss_fn2, has_aux=True)

    (l, pred), grads = grad_fn(state.params)
    return grad_fn, grads, l, loss_fn2, pred


@app.cell(hide_code=True)
def __():
    # def fill_template(template, pkey, fill_values):
    #     template = template.copy()
    #     """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we assign to the parameter
    #     """
    #     for i, k in enumerate(pkey):
    #         template["params"][k] = fill_values[i]
    #     return template


    # target_param = "_dawdreamer/freq"
    # param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 300, endpoint=False))
    # programs = [
    #     fill_template(copy.deepcopy(instrument_params), [target_param], [x])
    #     for x in param_linspace

    # ]
    # programs
    return


@app.cell
def __(copy, instrument_params, jnp, true_params):
    # make array of programs
    granularity = 10
    programs = [
        copy.deepcopy(instrument_params)
        for i in range(granularity ^ len(true_params))
    ]
    for k in instrument_params["params"].keys():
        vs = jnp.linspace(-1, 1, granularity, endpoint=False)
        for i, v in enumerate(vs):
            programs[i]["params"][k] = v
    return granularity, i, k, programs, v, vs


@app.cell
def __(programs):
    programs[0]
    return


@app.cell
def __(np):
    import pandas as pd

    x = np.linspace(-1, 1, 3)
    y = np.linspace(0, 1, 2)
    z = np.linspace(0, 1, 2)
    (
        xx,
        yy,
    ) = np.meshgrid(x, y, z)
    xx, yy
    return pd, x, xx, y, yy, z


@app.cell
def __(np, xx, yy, zz):
    np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    return


@app.cell
def __():
    # make programs using pandas
    return


@app.cell
def __():
    # p = instrument.apply(instrument_params, noise, SAMPLE_RATE)
    # l = naive_loss(spec_func(p)[0],target_spec)
    return


if __name__ == "__main__":
    app.run()

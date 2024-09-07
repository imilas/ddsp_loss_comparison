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
    import pandas as pd

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
        pd,
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
    cosine_distance = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # Spec function
    NFFT = 512
    WIN_LEN = 600
    HOP_LEN = 100
    spec_func = loss_helpers.spec_func(NFFT, WIN_LEN, HOP_LEN)


    def clip_spec(x):
        return jnp.clip(x, a_min=0, a_max=1)


    dtw_jax = softdtw_jax.SoftDTW(gamma=0.8)

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
def __(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)


    true_params = {"amp": 4,"carrier":70}
    init_params = {"amp": 2.5,"carrier":60}

    program = """
    import("stdfaust.lib");
    carrier = hslider("carrier",{carrier},20,300,0.1);
    amp = hslider("amp",{amp},0,10,0.1);
    sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
    process = sineOsc(amp)*sineOsc(carrier);
    """


    true_code = program.format(**true_params)
    instrument_code = program.format(**init_params)

    true_instrument, true_instrument_jit, true_noise, true_instrument_params = fj.code_to_flax(true_code, key)
    instrument, instrument_jit, noise, instrument_params = fj.code_to_flax(instrument_code, key)
    print(instrument_params)
    return (
        init_params,
        instrument,
        instrument_code,
        instrument_jit,
        instrument_params,
        key,
        noise,
        program,
        true_code,
        true_instrument,
        true_instrument_jit,
        true_instrument_params,
        true_noise,
        true_params,
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
def __(
    SAMPLE_RATE,
    dtw_jax,
    init_params,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    jnp,
    kernel,
    noise,
    np,
    onset_1d,
    optax,
    spec_func,
    target_sound,
    train_state,
    true_params,
):
    learning_rate = 0.03
    # Create Train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )


    # loss fn shows the difference between the output of synth and a target_sound
    def loss_fn(params):
        pred = instrument_jit(params, noise, SAMPLE_RATE)[0]
        # L1 time-domain loss
        # loss = (jnp.abs(pred - target_sound)).mean()
        # loss = naive_loss(spec_func(pred)[0],spec_func(target_sound))
        # loss = 1/dm_pix.ssim(clip_spec(spec_func(target_sound)),clip_spec(spec_func(pred)))
        # loss = dm_pix.simse(clip_spec(spec_func(target_sound)),clip_spec(spec_func(pred)))
        # loss = dtw_jax(pred,target_sound)
        loss = dtw_jax(onset_1d(target_sound, kernel, spec_func),onset_1d(pred, kernel, spec_func))
        # loss = naive_loss(scat_jax(target_sound), scat_jax(pred))
        # jax.debug.print("loss_helpers:{y},loss:{l}",y=loss_helpers.onset_1d(pred)[0:3],l=loss)
        return loss, pred

    # Clip gradients function
    def clip_grads(grads, clip_norm):
        total_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(grads)))
        scale = clip_norm / jnp.maximum(total_norm, clip_norm)
        return jax.tree_util.tree_map(lambda g: g * scale, grads)


    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


    @jax.jit
    def train_step(state):
        """Train for a single step."""
        # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        grads = clip_grads(grads, clip_norm=1.0)
        state = state.apply_gradients(grads=grads)
        return state, loss


    losses = []
    sounds = []
    real_params = {k: [init_params[k]] for k in true_params.keys()}  # will record parameters while searching
    norm_params = {k: [] for k in true_params.keys()}  # will record parameters while searching

    for n in range(150):
        state, loss = train_step(state)
        if n % 1 == 0:
            audio, mod_vars = instrument_jit(state.params, noise, SAMPLE_RATE)
            sounds.append(audio)
            for pname in real_params.keys():
                parameter_value = np.array(
                    mod_vars["intermediates"]["dawdreamer/%s" % pname]
                )[0]
                real_params[pname].append(parameter_value)
                norm_params[pname].append(state.params["params"]["_dawdreamer/%s" % pname])
            losses.append(loss)
            # print(n, loss, state.params)
            print(n, end="\r")
    return (
        audio,
        clip_grads,
        grad_fn,
        learning_rate,
        loss,
        loss_fn,
        losses,
        mod_vars,
        n,
        norm_params,
        parameter_value,
        pname,
        real_params,
        sounds,
        state,
        train_step,
        tx,
    )


@app.cell
def __(losses, mo, plt, real_params, true_params):
    mo.output.clear()
    fig1, axs = plt.subplots(1, len(real_params) + 1, figsize=(12, 3))
    axs[0].set_xlabel("time (s)")
    axs[0].set_ylabel("loss", color="black")
    axs[0].plot(losses, color="black")

    c = 0
    colors = ["red", "green", "blue", "purple"]
    for pname2, pvalue in real_params.items():
        ax = axs[c + 1]
        ax.set_ylabel(pname2)  # we already handled the x-label with ax1
        ax.plot(real_params[pname2], color=colors[c])
        ax.axhline(true_params[pname2], linestyle="dashed", color=colors[c], label="true")
        ax.tick_params(axis="y")
        c += 1
    fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    fig1
    return ax, axs, c, colors, fig1, pname2, pvalue


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
def __(instrument_params, np, pd):


    def make_programs_df(true_params, granularity):
        # make programs that cover the grid, given the parameters dict
        meshgrid = np.meshgrid(*[np.linspace(-0.95, 1, granularity, endpoint=False) for i in range(len(true_params.keys()))])
        program_params = np.stack([d.flatten() for d in meshgrid], axis=-1)
        programs_df = pd.DataFrame(program_params, columns=true_params.keys())
        return programs_df


    granularity = 8
    programs_df = make_programs_df(instrument_params["params"], granularity)
    return granularity, make_programs_df, programs_df


@app.cell
def __(grad_fn, programs_df):
    grad_loss_dict = [grad_fn({"params": programs_df.loc[i].to_dict()}) for i in range(len(programs_df))]
    return grad_loss_dict,


@app.cell
def __(
    grad_loss_dict,
    granularity,
    instrument_params,
    mo,
    norm_params,
    np,
    plt,
    programs_df,
    true_instrument_params,
):
    mo.output.clear()
    plt.figure(figsize=(6, 6))
    data = np.array([x[0][0] for x in grad_loss_dict]).reshape(granularity, granularity)

    plt.imshow(data, cmap="coolwarm", extent=(-1, 1, -1, 1), origin="lower")

    grad_dir = np.array([list(x[1]["params"].values()) for x in grad_loss_dict])
    grad_dir = np.where(grad_dir>0,1,-1)
    plt.quiver(*programs_df.T.to_numpy(), *-grad_dir.T)
    plt.scatter(*true_instrument_params["params"].values(), color="#00FF00", marker="*", s=[150])
    plt.scatter(*instrument_params["params"].values(), color="#00FF00", marker="o", s=[150])

    # path 
    plt.scatter(*list(norm_params.values()),color="white",alpha=0.5,s=[30],marker=".")

    #
    plt.xlabel(programs_df.columns[0])
    plt.ylabel(programs_df.columns[1])
    return data, grad_dir


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    # ideas:
    # show path traveled on quiver plots
    # fix dtw_loss
    # fix multi-level spec loss
    # try the modulated sine cut-off program in original code
    # make a grad surface and calculate how useful a loss is with how many points it can reach the goal based on a learning rate
    # multi-objective loss function
    #
    # have to define your loss functions 
    # have to define your programs (maybe before your loss functions so you can show landscapes)
    return


if __name__ == "__main__":
    app.run()

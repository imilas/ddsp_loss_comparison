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
    from audax.core import functional
    import copy
    import dm_pix

    from helpers import faust_to_jax as fj
    from helpers import onsets
    from helpers.softdtw_jax import SoftDTW


    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be


    naive_loss = lambda x, y: jnp.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

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
        normalized=True,
        center=True,
        onesided=True,
    )
    fb = functional.melscale_fbanks(
        n_freqs=(NFFT // 2) + 1,
        n_mels=64,
        sample_rate=SAMPLE_RATE,
        f_min=60.0,
        f_max=10000.0,
    )


    def clipped_spec(x):
        jax_spec = spec_func(x)
        jax_spec = jnp.clip(jax_spec, a_min=0, a_max=1)
        return jax_spec
        
    return (
        HOP_LEN,
        NFFT,
        Path,
        SAMPLE_RATE,
        SoftDTW,
        WIN_LEN,
        clipped_spec,
        copy,
        cosine_distance,
        default_device,
        dm_pix,
        fb,
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
        onsets,
        optax,
        os,
        partial,
        plt,
        spec_func,
        train_state,
        unfreeze,
        wavfile,
        window,
    )


@app.cell
def __(mo):
    mo.md(
        """
    There are 2 types of parameter search problems, inside search, and outside search. 

    - Inside search is easier: we create a target sound using the synth, then we randomly initialize the synthesizer parameters and search for the target parameters. We know that the synth can create an output that is (at least to our ears), identical to the target sound
    - Outside search means that the target sound is from a source other than the synthesizer. We don't know how close the synthesizer can get to the target sound. This also means there is a degree of subjectivity to outside search. 

    Here we focus on inside search: 

    Each program has 1 or 2 parameters. Each parameter has a TRUE value which the search attemps to find, and and INIT value that the parameter is initially set to. 
    We conduct stochastic gradient descent search using various loss functions and plot the change in parameters
    """
    )
    return


@app.cell
def __(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    # faust_code_target = f"""
    # import("stdfaust.lib");
    # cutoff = hslider("freq", 4000, 20., 20000., .01);
    # process = fi.lowpass(1, cutoff);
    # """

    # faust_code_instrument = f"""
    # import("stdfaust.lib");
    # cutoff = hslider("freq", 10000, 20., 20000., .01);
    # process = fi.lowpass(1, cutoff);
    #  """

    true_params = {"lp_cut": 3000, "hp_cut": 2000}
    init_params = {"lp_cut": 10000, "hp_cut": 500}
    faust_code_target = f"""
    import("stdfaust.lib");
    lp_cut = hslider("lp_cut", {true_params["lp_cut"]}, 2000., 20000., .01);
    hp_cut = hslider("hp_cut", {true_params["hp_cut"]}, 20., 5000., .01);
    process = fi.lowpass(5, lp_cut):fi.highpass(5,hp_cut);
    """
    faust_code_instrument = f"""
    import("stdfaust.lib");
    lp_cut = hslider("lp_cut", {init_params["lp_cut"]}, 2000., 20000., .01);
    hp_cut = hslider("hp_cut",{true_params["hp_cut"]}, 20., 5000., .01);
    process = fi.lowpass(5, lp_cut):fi.highpass(5,hp_cut);
     """

    # faust_code_target = f"""
    # import("stdfaust.lib");
    # osc_f = hslider("freq",3,1,20,0.5);
    # cutoff = hslider("cutoff",1000,101,1000,1);
    # FX = fi.lowpass(5,cutoff);
    # process = os.osc(os.osc(osc_f)*4)*400,_:["cutoff":+(_,_)->FX];
    # """

    # faust_code_instrument = f"""
    # import("stdfaust.lib");
    # osc_f = hslider("freq",20,1,20,0.5);
    # cutoff = hslider("cutoff",10000,101,20000,1);
    # FX = fi.lowpass(5,cutoff);
    # process = os.osc(os.osc(osc_f)*4)*400,_:["cutoff":+(_,_)->FX];
    # """
    return (
        faust_code_instrument,
        faust_code_target,
        init_params,
        key,
        true_params,
    )


@app.cell
def __():
    return


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
    print(instrument_params)
    return init_sound, target_sound


@app.cell
def __(fb, functional, partial, plt, spec_func, target_sound):

    mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)
    target_spec = spec_func(target_sound)[0]
    plt.imshow(target_spec)
    return mel_spec_func, target_spec


@app.cell
def __(init_sound, plt, spec_func):
    plt.imshow(spec_func(init_sound)[0])
    return


@app.cell
def __(mo):
    mo.md(
        """
    Let's setup an SGD experiment with a customizable loss function. 
    We try:
        - Naive loss 
        - Naive loss with spectrogram
        - SSIM loss (gets close to correct )
        - SIMSE loss (Finds correct parameter)
    """
    )
    return


@app.cell
def __(SoftDTW, init_sound, jax, onsets, target_sound):
    # strctural similarty measure

    # dm_pix.ssim(clipped_spec(target_sound), clipped_spec(target_sound))

    dtw_jax = SoftDTW(gamma=0.01)
    dtw_jit = jax.jit(dtw_jax)

    dtw_jit(onsets.onset_1d(target_sound),onsets.onset_1d(init_sound))

    return dtw_jax, dtw_jit


@app.cell
def __(
    SAMPLE_RATE,
    dtw_jit,
    init_params,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    jnp,
    noise,
    np,
    onsets,
    optax,
    target_sound,
    train_state,
    true_params,
):
    learning_rate = 0.0001
    # Create Train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )


    # loss fn shows the difference between the output of synth and a target_sound
    def loss_fn(params):
        pred = instrument.apply(params, noise, SAMPLE_RATE)
        # L1 time-domain loss
        # loss = (jnp.abs(pred - target_sound)).mean()
        # loss = naive_loss(spec_func(pred)[0],target_spec)
        # loss = 1/dm_pix.ssim(clipped_spec(target_sound),clipped_spec(pred))
        # loss = dm_pix.simse(clipped_spec(target_sound),clipped_spec(pred))

        loss = jnp.abs(1/dtw_jit(onsets.onset_1d(target_sound),onsets.onset_1d(pred)))
        return loss, pred


    # @jax.jit
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
    for n in range(10):
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
            print(n, loss, state.params)
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
def __():
    # search_params["lp_cut"].append(1)
    # search_params
    return


@app.cell
def __(losses, plt, search_params, true_params):
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
def __(plt, sounds, spec_func):
    plt.imshow(spec_func(sounds[-1])[0])
    return


@app.cell
def __(fj, sounds):
    fj.show_audio(sounds[-1])
    return


@app.cell
def __(mo):
    mo.md(
        """We see that the spectrum loss fails at finding the parameter because we did't use the same instance of noise when creating our target sound vs when we were searching for the filter parameters. Below we should that two different noises fed through the filter will look different in our spectrum loss function even though they sound the same"""
    )
    return


@app.cell
def __(
    SAMPLE_RATE,
    clipped_spec,
    dm_pix,
    fj,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    length_seconds,
    naive_loss,
    spec_func,
):
    # instrument_jit(instrument_params, noise, SAMPLE_RATE)[0]
    noise_1 = jax.random.uniform(
        jax.random.PRNGKey(10),
        [instrument.getNumInputs(), SAMPLE_RATE * length_seconds],
        minval=-1,
        maxval=1,
    )
    noise_2 = jax.random.uniform(
        jax.random.PRNGKey(3),
        noise_1.shape,
        minval=-1,
        maxval=1,
    )
    processed_noise_1 = instrument_jit(instrument_params, noise_1, SAMPLE_RATE)[0]
    processed_noise_2 = instrument_jit(instrument_params, noise_2, SAMPLE_RATE)[0]


    print(
        " spec loss value:",
        naive_loss(spec_func(processed_noise_1), spec_func(processed_noise_2)),
    )

    print(
        "ssim loss value:",
        1
        / dm_pix.ssim(
            clipped_spec(processed_noise_1), clipped_spec(processed_noise_2)
        ),
    )


    fj.show_audio(processed_noise_1)
    fj.show_audio(processed_noise_2)
    return noise_1, noise_2, processed_noise_1, processed_noise_2


if __name__ == "__main__":
    app.run()

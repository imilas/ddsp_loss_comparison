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
    import optax

    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt

    # import matplotlib.animation as animation
    # from matplotlib import rc

    # from IPython.display import HTML
    # from IPython.display import Audio
    # import IPython.display as ipd

    from helpers import faust_to_jax as fj
    from audax.core import functional
    import copy
    from helpers import ts_comparisions as ts_comparisons
    import dtw

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    return (
        Path,
        SAMPLE_RATE,
        copy,
        default_device,
        dtw,
        fj,
        functional,
        functools,
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
        ts_comparisons,
        wavfile,
    )


@app.cell
def __(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code_1 = """
    import("stdfaust.lib");
    cutoff = hslider("cutoff", 1000., 20., 20000., .01);
    process = fi.lowpass(10, cutoff);
    """

    faust_code_2 = """
    import("stdfaust.lib");
    cutoff = hslider("cutoff",500,101,1000,1);
    osc_f = hslider("osc_f",10,1,20,0.5);
    //osc_mag = hslider("osc_mag",200,10,400,1);
    FX = fi.lowpass(5,cutoff);
    process = os.osc(osc_f)*400,_:["cutoff":+(_,_)->FX];
    """

    faust_code_3 = """
    import("stdfaust.lib");
    cutoff = hslider("cutoff",500,101,1000,1);
    osc_f = hslider("osc_f",3,1,20,0.5);
    FX = fi.lowpass(5,cutoff);
    process = os.osc(os.osc(osc_f)*4)*400,_:["cutoff":+(_,_)->FX];
    """
    return faust_code_1, faust_code_2, faust_code_3, key


@app.cell
def __(SAMPLE_RATE, faust_code_3, fj, jax, key):
    DSP = fj.faust2jax(faust_code_3)
    DSP = DSP(SAMPLE_RATE)
    DSP_jit = jax.jit(DSP.apply, static_argnums=[2])
    noise = jax.random.uniform(
        jax.random.PRNGKey(10),
        [DSP.getNumInputs(), SAMPLE_RATE],
        minval=-1,
        maxval=1,
    )
    DSP_params = DSP.init(key, noise, SAMPLE_RATE)
    return DSP, DSP_jit, DSP_params, noise


@app.cell
def __(faust_code_3, fj, key, mo):
    mo.output.clear()
    target, _ = fj.process_noise_in_faust(faust_code_3, key)
    fj.show_audio(target)
    return target,


@app.cell
def __(mo):
    mo.md(
        """let's explore the loss landscape while modifying the osc_frequency.

    We make a target sound, which is the output of a faust program, and compare the loss/difference with the outputs of several 
    faust synthesizer programs. Each program has a unique osc_frequency value. 

    We try naive loss, spectrogram loss, wavelet loss
    """
    )
    return


@app.cell
def __(DSP_jit, DSP_params, SAMPLE_RATE, copy, jnp, noise):
    def fill_template(template, pkey, fill_values):
        template = template.copy()
        """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we assign to the parameter
        """
        for i, k in enumerate(pkey):
            template["params"][k] = fill_values[i]
        return template


    param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 300, endpoint=False))
    programs = [
        fill_template(copy.deepcopy(DSP_params), ["_dawdreamer/osc_f"], [x])
        for x in param_linspace
    ]

    outputs = [DSP_jit(p, noise, SAMPLE_RATE)[0] for p in programs]
    return fill_template, outputs, param_linspace, programs


@app.cell
def __(DSP_params, np, outputs, param_linspace, plt, target):
    naive_loss = lambda x, y: np.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    losses_naive = [naive_loss(x, target[0]) for x in outputs]
    plt.plot(param_linspace, losses_naive)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("naive loss on signals")
    return cosine_distance, losses_naive, naive_loss


@app.cell
def __(mo):
    mo.md(
        """
    ### spectrogram loss
    """
    )
    return


@app.cell
def __(SAMPLE_RATE, functional, jnp, outputs, partial, target):
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
        f_max=7800.0,
    )


    mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)
    target_spec = spec_func(target)[0].T
    output_specs = [spec_func(x)[0].T for x in outputs]
    return (
        HOP_LEN,
        NFFT,
        WIN_LEN,
        fb,
        mel_spec_func,
        output_specs,
        spec_func,
        target_spec,
        window,
    )


app._unparsable_cell(
    r"""
        losses_spec = [naive_loss(x, target_spec).mean() for x in output_specs]
        plt.plot(param_linspace, losses_spec)
        plt.axvline(
            DSP_params[\"params\"][\"_dawdreamer/osc_f\"],
            color=\"#FF000055\",
            linestyle=\"dashed\",
            label=\"correct param\",
        )
        plt.legend()
        plt.title(\"naive loss with spectrograms\")
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        """
    ### structural similarity measure.
    https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    """
    )
    return


@app.cell
def __(
    DSP_params,
    jax,
    jnp,
    outputs,
    param_linspace,
    plt,
    spec_func,
    target,
):
    from skimage.metrics import structural_similarity as ssim
    from skimage import img_as_float


    @jax.jit
    def clipped_spec(x):
        jax_spec = spec_func(x)
        jax_spec = jnp.clip(jax_spec, a_min=0, a_max=1)
        return jax_spec


    tsc = clipped_spec(target)
    clipped_specs = [clipped_spec(x) for x in outputs]
    losses_ssim = [
        ssim(img_as_float(x[0]), img_as_float(tsc[0]), data_range=1)
        for x in clipped_specs
    ]
    plt.plot(param_linspace, losses_ssim)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("structural similarity measure of spectrograms")
    return clipped_spec, clipped_specs, img_as_float, losses_ssim, ssim, tsc


@app.cell
def __(mo):
    mo.md(
        """
    # Pre-trained models
    Let's try out leaf, and see if it's any different from spectrograms. 
    """
    )
    return


@app.cell
def __(
    DSP_params,
    SAMPLE_RATE,
    jax,
    jnp,
    key,
    naive_loss,
    outputs,
    param_linspace,
    plt,
    target,
):
    from audax.frontends import leaf

    leaf = leaf.Leaf(sample_rate=SAMPLE_RATE, min_freq=30, max_freq=20000)
    leaf_params = leaf.init(key, target)
    leaf_apply = jax.jit(leaf.apply)
    target_leaf = leaf_apply(leaf_params, target)
    output_leafs = [
        leaf_apply(leaf_params, jnp.expand_dims(x, axis=0)) for x in outputs
    ]
    losses_leaf = [naive_loss(x[0], target_leaf).mean() for x in output_leafs]
    plt.plot(param_linspace, losses_leaf)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("naive loss with leaf specs")
    return (
        leaf,
        leaf_apply,
        leaf_params,
        losses_leaf,
        output_leafs,
        target_leaf,
    )


@app.cell
def __(
    DSP_params,
    SAMPLE_RATE,
    jax,
    jnp,
    key,
    naive_loss,
    outputs,
    param_linspace,
    plt,
    target,
):
    from audax.frontends import sincnet

    snet = sincnet.SincNet(sample_rate=SAMPLE_RATE)
    snet_params = snet.init(key, target)
    snet_apply = jax.jit(snet.apply)
    target_snet = snet_apply(snet_params, target)
    output_snet = [
        snet_apply(snet_params, jnp.expand_dims(x, axis=0)) for x in outputs
    ]
    losses_snet = [naive_loss(x[0], target_snet).mean() for x in output_snet]
    plt.plot(param_linspace, losses_snet)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("naive loss with snet specs")
    return (
        losses_snet,
        output_snet,
        sincnet,
        snet,
        snet_apply,
        snet_params,
        target_snet,
    )


@app.cell
def __(mo):
    mo.md(
        """Let's try non spectrogram features:
        We calculate the onset values, which show when in the spectrogram music "events" seem to be happening.
        We then use CBD(compression based similariy) and dtw_losses to see if the onset of musical events matches. 
        This gives us loss landscapes that are much more convex than spectrogram differences. 


    """
    )
    return


@app.cell
def __(SAMPLE_RATE, librosa, np, outputs, target):
    output_onsets = [
        librosa.onset.onset_strength_multi(
            y=np.array(y), sr=SAMPLE_RATE, channels=[0, 32, 64, 96, 128]
        )
        for y in outputs
    ]
    target_onset = librosa.onset.onset_strength_multi(
        y=np.array(target), sr=SAMPLE_RATE, channels=[0, 32, 64, 96, 128]
    )
    return output_onsets, target_onset


@app.cell
def __(
    DSP_params,
    np,
    output_onsets,
    param_linspace,
    plt,
    target_onset,
    ts_comparisons,
):
    # we calculate the onsets then use cbd loss

    cbd = ts_comparisons.CompressionBasedDissimilarity()

    cbd_loss = [
        cbd.calculate(
            np.array(target_onset[0]).sum(axis=0), np.array(x).sum(axis=0)
        )
        for x in output_onsets
    ]

    plt.plot(param_linspace, cbd_loss)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("cbd loss using onsets")
    return cbd, cbd_loss


@app.cell
def __(
    DSP_params,
    dtw,
    np,
    output_onsets,
    param_linspace,
    plt,
    target_onset,
):
    def dtw_loss(x1, x2):

        query = np.array(x1).sum(axis=0)
        template = np.array(x2).sum(axis=0)
        alignment = dtw.dtw(
            query,
            template,
            keep_internals=True,
            step_pattern=dtw.rabinerJuangStepPattern(6, "c"),
        )
        return alignment.normalizedDistance


    dtw_losses = [dtw_loss(target_onset[0], x) for x in output_onsets]

    plt.plot(param_linspace, dtw_losses)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("dtw loss using onsets")
    return dtw_loss, dtw_losses


@app.cell
def __():
    # todo:
    # - test out another program or two and show whether loss function is useful
    # - use loss functions to train a network
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

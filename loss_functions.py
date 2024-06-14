import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __(boxFromDSP, boxToSource):
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
    import dawdreamer.faust.box as box

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

    SAMPLE_RATE = 44100


    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.0:
            data /= abs(data).max()

        ipd.display(
            Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)
        )


    def faust2jax(faust_code: str):
        """
        Convert faust code into a batched JAX model and a single-item inference function.

        Inputs:
        * faust_code: string of faust code.
        """

        module_name = "MyDSP"
        with FaustContext():

            box = boxFromDSP(faust_code)

            jax_code = boxToSource(
                box, "jax", module_name, ["-a", "jax/minimal.py"]
            )

        custom_globals = {}

        exec(jax_code, custom_globals)  # security risk!

        MyDSP = custom_globals[module_name]
    return (
        Audio,
        FaustContext,
        HTML,
        Path,
        SAMPLE_RATE,
        Scattering1D,
        animation,
        box,
        createLibContext,
        default_device,
        destroyLibContext,
        faust2jax,
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
        show_audio,
        time,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(jax):
    from helpers import faust_to_jax as fj
    from audax.core import functional
    import copy
    # from kymatio.numpy import Scattering1D

    SAMPLE_RATE = 44100
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
    return (
        SAMPLE_RATE,
        copy,
        faust_code_1,
        faust_code_2,
        fj,
        functional,
        key,
    )


@app.cell
def __(SAMPLE_RATE, faust_code_2, fj, jax, key):
    DSP = fj.faust2jax(faust_code_2)
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
def __(faust_code_2, fj, key, mo):
    mo.output.clear()
    target, _ = fj.process_noise_in_faust(faust_code_2, key)
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
        """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we want to change
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
    cosine_distance = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

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


@app.cell(hide_code=True)
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


@app.cell
def __(
    DSP_params,
    naive_loss,
    output_specs,
    param_linspace,
    plt,
    target_spec,
):
    losses_spec = [naive_loss(x, target_spec).mean() for x in output_specs]
    plt.plot(param_linspace, losses_spec)
    plt.axvline(
        DSP_params["params"]["_dawdreamer/osc_f"],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("naive loss with spectrograms")
    return losses_spec,


@app.cell
def __(mo):
    mo.md('''
    ### structural similarity measure.
    https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    ''')
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
    losses_ssim = [ssim(img_as_float(x[0]),img_as_float(tsc[0]),data_range=1) for x in clipped_specs ]
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
    mo.md('''
    # Pre-trained models
    Let's try out leaf, and see if it's any different from spectrograms. 
    ''')
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

    leaf = leaf.Leaf(sample_rate=SAMPLE_RATE,min_freq=30,max_freq=20000)
    leaf_params = leaf.init(key,target)
    leaf_apply = jax.jit(leaf.apply)
    target_leaf = leaf_apply(leaf_params,target)
    output_leafs = [leaf_apply(leaf_params,jnp.expand_dims(x,axis=0)) for x in outputs]
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
    snet_params = snet.init(key,target)
    snet_apply = jax.jit(snet.apply)
    target_snet = snet_apply(snet_params,target)
    output_snet = [snet_apply(snet_params,jnp.expand_dims(x,axis=0)) for x in outputs]
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
    mo.md('''
    Let's try non spectrogram features

    ''')
    return


@app.cell
def __():
    from helpers import librosa_features as lf

    lf.extract_feature_means('sounds/target_sounds/two_osc_396_5.5.wav')
    return lf,


if __name__ == "__main__":
    app.run()

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
    import copy

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
    return SAMPLE_RATE, copy, faust_code_1, faust_code_2, fj, key


@app.cell
def __(SAMPLE_RATE, faust_code_2, fj, jax, key):
    DSP = fj.faust2jax(faust_code_2)
    DSP = DSP(SAMPLE_RATE)
    DSP_jit = jax.jit(DSP.apply,static_argnums=[2])
    noise = jax.random.uniform(jax.random.PRNGKey(10),[DSP.getNumInputs(),SAMPLE_RATE],minval=-1,maxval=1)
    DSP_params = DSP.init(key,noise,SAMPLE_RATE)
    return DSP, DSP_jit, DSP_params, noise


@app.cell
def __(DSP_jit, DSP_params, SAMPLE_RATE, noise):
    output = DSP_jit(DSP_params,noise,SAMPLE_RATE,)
    return output,


@app.cell
def __():
    # let's explore the loss landscape while modifying the osc_frequency

    return


@app.cell
def __(DSP_jit, DSP_params, SAMPLE_RATE, copy, fj, jnp, noise):
    def fill_template(template,pkey,fill_values):
        template = template.copy()
        '''template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we want to change
        '''
        for i,k in enumerate(pkey):
            template["params"][k] = fill_values[i]
        return template

    programs = [fill_template(copy.deepcopy(DSP_params),["_dawdreamer/osc_f"],[x]) for x in jnp.array(jnp.linspace(-1.,1.,10))] 

    outputs = [DSP_jit(p,noise,SAMPLE_RATE)[0] for p in programs]

    for o in outputs:
        fj.show_audio(o)

    return fill_template, o, outputs, programs


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

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

SAMPLE_RATE = 44100

def show_audio(data, autoplay=False):
    if abs(data).max() > 1.0:
        data /= abs(data).max()

    ipd.display(
        Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)
    )


def faust2jax(faust_code: str,module_name="MyDsp"):
    """
    Inputs:
    * faust_code: string of faust code.
    """

    with FaustContext():

        box = fbox.boxFromDSP(faust_code)

        jax_code = fbox.boxToSource(
            box, "jax", module_name, ["-a", "jax/minimal.py"]
        )

    custom_globals = {}

    exec(jax_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]
    return MyDSP

def process_noise_in_faust(faust_code, key,length_seconds=1):
    key, subkey = jax.random.split(key)
    DSP = faust2jax(faust_code)
    lp = DSP(SAMPLE_RATE)  # init model
    noise = jax.random.uniform(
        subkey, [DSP.getNumInputs(DSP), SAMPLE_RATE*length_seconds], minval=-1, maxval=1
    )  # make input
    lp_params = lp.init(subkey, noise, SAMPLE_RATE)
    return lp.apply(lp_params, noise, SAMPLE_RATE), subkey

def code_to_flax(synth_code,key,length_seconds=1):
    """
    Inputs: faust code
    Outputs: jax instrument, jitted instrument, instrument input (still needed if empty), and instrument flax params
    
    """
    DSP = faust2jax(synth_code)
    instrument = DSP(SAMPLE_RATE)  # init model
    noise = jax.random.uniform(
        key,
        [instrument.getNumInputs(), SAMPLE_RATE * length_seconds],
        minval=-1,
        maxval=1,
    )
    instrument_params = instrument.init(key, noise, SAMPLE_RATE)
    instrument_jit = jax.jit(
        partial(instrument.apply, mutable="intermediates"), static_argnums=[2]
    )
    return instrument, instrument_jit, noise, instrument_params

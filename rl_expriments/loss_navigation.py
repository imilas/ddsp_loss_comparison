import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    _parentdir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(_parentdir))


    import marimo as mo
    import functools
    from functools import partial
    import itertools
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

    from helper_funcs import program_generators as pg
    from helper_funcs import faust_to_jax as fj
    from helper_funcs import loss_helpers
    from helper_funcs import softdtw_jax
    from helper_funcs.experiment_scripts import append_to_json
    from helper_funcs.program_generators import choose_program, generate_parameters
    import random

    from kymatio.jax import Scattering1D
    import json
    import argparse
    from helper_funcs import experiment_setup as setup

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return (
        SAMPLE_RATE,
        argparse,
        fj,
        jax,
        length_seconds,
        mo,
        partial,
        pg,
        setup,
    )


@app.cell
def _(argparse, setup):
    parser = argparse.ArgumentParser(description='Process a loss function name.')
    parser.add_argument('--loss_fn', type=str, help='the name of the loss function. One of:  L1_Spec , DTW_Onset, SIMSE_Spec, JTFS',default="L1_Spec")
    parser.add_argument('--learning_rate', type=float, help='learning rate',default=0.01)
    parser.add_argument('--program_id', type=int, choices=[0, 1, 2, 3], default = 0, help="The program ID to select (0, 1, 2, or 3)")
    args, unknown = parser.parse_known_args()
    spec_func = setup.spec_func
    clip_spec = setup.clip_spec
    naive_loss = setup.naive_loss
    dtw_jax = setup.dtw_jax
    scat_jax = setup.scat_jax
    kernel = setup.kernel
    onset_1d = setup.onset_1d
    return


@app.cell
def _():
    # i am making fause programs, which can have a number of sliders.
    # each slider has a default value, and a range (min,max). The range is useful to set boundries for parameter updates
    # let's say the function that generates these programs is called generate_program.
    # each program could be a json file that has an id and program code with placeholders for the parameters (parameters being valid ranges and values)
    # generate_program takes in the program id, a range, and optionally a set of values for each parameter. If the values are not defined, it will arbitrarly chooose a value between the range for each parameter. generate_program then returns the completed program, and a dictionary of parameter names and ranges. 
    return


@app.cell
def _():
    return


@app.cell
def _(SAMPLE_RATE, fj, jax, pg):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code,_ = pg.generate_lp_1D([100,1000])
    print(faust_code)
    return faust_code, key


@app.cell
def _(SAMPLE_RATE, faust_code, fj, jax, key, length_seconds, partial):
    DSP = fj.faust2jax(faust_code)
    DSP = DSP(SAMPLE_RATE)
    DSP_jit = jax.jit(partial(DSP.apply,mutable="intermediates"), static_argnums=[1])
    noise = jax.random.uniform(
        jax.random.PRNGKey(10),
        [DSP.getNumInputs(), SAMPLE_RATE * length_seconds],
        minval=0,
        maxval=0,
    )
    DSP_params = DSP.init(key, noise, SAMPLE_RATE)
    return (DSP_params,)


@app.cell
def _(DSP_params):
    DSP_params
    return


@app.cell
def _(faust_code, fj, key, length_seconds, mo):
    mo.output.clear()
    target, _ = fj.process_noise_in_faust(
        faust_code, key, length_seconds=length_seconds
    )
    fj.show_audio(target)
    return


if __name__ == "__main__":
    app.run()

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
        copy,
        dm_pix,
        fj,
        jax,
        jnp,
        length_seconds,
        mo,
        np,
        partial,
        pg,
        plt,
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
    return (
        clip_spec,
        dtw_jax,
        kernel,
        naive_loss,
        onset_1d,
        scat_jax,
        spec_func,
    )


@app.cell
def _():
    # i am making fause programs, which can have a number of sliders.
    # each slider has a default value, and a range (min,max). The range is useful to set boundries for parameter updates
    # let's say the function that generates these programs is called generate_program.
    # each program could be a json file that has an id and program code with placeholders for the parameters (parameters being valid ranges and values)
    # generate_program takes in the program id, a range, and optionally a set of values for each parameter. If the values are not defined, it will arbitrarly chooose a value between the range for each parameter. generate_program then returns the completed program, and a dictionary of parameter names and ranges. 
    return


@app.cell
def _(SAMPLE_RATE, fj, jax, pg):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code,_ = pg.generate_lp_1D([100,1000])
    # faust_code,_ = pg.generate_program_2_1D([0.1,20])
    print(faust_code)
    return faust_code, key


@app.cell
def _(SAMPLE_RATE, faust_code, fj, jax, key, partial):
    DSP = fj.faust2jax(faust_code)
    DSP = DSP(SAMPLE_RATE)
    instrument_jit = jax.jit(partial(DSP.apply,mutable="intermediates"), static_argnums=[2])
    noise = jax.random.uniform(
        jax.random.PRNGKey(20),
        [DSP.getNumInputs(), SAMPLE_RATE],
        minval=-1,
        maxval=1,
    )
    DSP_params = DSP.init(key, noise, SAMPLE_RATE)
    print(DSP_params)
    return DSP_params, instrument_jit, noise


@app.cell
def _(faust_code, fj, key, length_seconds, mo):
    mo.output.clear()
    target_sound, _ = fj.process_noise_in_faust(
        faust_code, key, length_seconds=length_seconds
    )
    fj.show_audio(target_sound)
    return (target_sound,)


@app.cell
def _(DSP_params, SAMPLE_RATE, copy, fj, instrument_jit, jnp, noise):
    def fill_template(template, pkey, fill_values):
        template = template.copy()
        """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we assign to the parameter
        """
        for i, k in enumerate(pkey):
            template["params"][k] = fill_values[i]
        return template


    target_param = list(DSP_params["params"].keys())[0]
    param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 300, endpoint=False))
    programs = [
        fill_template(copy.deepcopy(DSP_params), [target_param], [x])
        for x in param_linspace
    ]

    s,_ = instrument_jit(programs[100], noise, SAMPLE_RATE)
    fj.show_audio(s)
    return param_linspace, programs


@app.cell
def _(
    SAMPLE_RATE,
    clip_spec,
    dm_pix,
    dtw_jax,
    instrument_jit,
    jax,
    kernel,
    loss_multi_spec,
    naive_loss,
    noise,
    onset_1d,
    programs,
    scat_jax,
    spec_func,
    target_sound,
):
    lfn = 'DTW_Onset'
    def loss_fn(params):
        pred = instrument_jit(params, noise, SAMPLE_RATE)[0]
        # loss = (jnp.abs(pred - target_sound)).mean()
        # loss = 1/dm_pix.ssim(clip_spec(spec_func(target_sound)),clip_spec(spec_func(pred)))
        if lfn  == 'L1_Spec':
            loss = naive_loss(spec_func(pred)[0], spec_func(target_sound))
        elif lfn  == 'SIMSE_Spec':
            loss = dm_pix.simse(clip_spec(spec_func(target_sound)), clip_spec(spec_func(pred)))
        elif lfn  == 'DTW_Onset':
            loss = dtw_jax(onset_1d(target_sound, kernel, spec_func), onset_1d(pred, kernel, spec_func))
        elif lfn  == 'JTFS':
            loss = naive_loss(scat_jax(target_sound), scat_jax(pred)[0])
        elif lfn == 'Multi_Spec':
            loss = loss_multi_spec(target_sound,pred)
        else:
            raise ValueError("Invalid value for loss")  
        return loss, pred


    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss, pred), grads = grad_fn(programs[0])
    return (grad_fn,)


@app.cell
def _(grad_fn, programs):
    lpg = [grad_fn(p) for p in programs]
    lpg[0]
    return (lpg,)


@app.cell
def _(lpg):
    g = [list(p[1]["params"].values())[0] for p in lpg]
    return (g,)


@app.cell
def _(DSP_params, g, lpg, np, param_linspace, plt):
    fig, ax1 = plt.subplots()

    # Plot on the first y-axis
    ax1.plot(param_linspace, [p[0][0] for p in lpg], label='Loss', color='blue')
    ax1.set_ylabel("Loss", color='blue')

    # Create second y-axis
    ax2 = ax1.twinx()
    eps = 1e-3  # or any small value
    result = [-1 if x < -eps else 1 if x > eps else 0 for x in g]
    window_size = 3
    window = np.ones(window_size) / window_size
    result = np.convolve(result, window, mode='same')
    # ax2.plot(param_linspace, g, label='Normalized g', color='green')
    ax2.plot(param_linspace, result, color='green' )

    ax2.set_ylabel("smoothed gradients", color='green')
    ax1.axvline(x=list(DSP_params["params"].values())[0], color='red', linestyle='--', linewidth=2)

    plt.show()
    return


if __name__ == "__main__":
    app.run()

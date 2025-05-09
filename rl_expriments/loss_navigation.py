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
    project_root = Path(__file__).resolve().parent.parent
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
        choose_program,
        dm_pix,
        fj,
        generate_parameters,
        jax,
        jnp,
        mo,
        np,
        optax,
        plt,
        setup,
        train_state,
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
        args,
        clip_spec,
        dtw_jax,
        kernel,
        naive_loss,
        onset_1d,
        scat_jax,
        spec_func,
    )


@app.cell
def _(args, choose_program, generate_parameters):
    var1_range, var2_range,true_var1,true_var2 = generate_parameters(1)
    rand_prog_code, var1_value, var2_value = choose_program(args.program_id, var1_range, var2_range)
    true_prog_code, true_var1_value, true_var2_value = choose_program(args.program_id,var1_range, var2_range,true_var1,true_var2)
    print("Program Code:\n", true_prog_code)
    print("init vars",var1_value,var2_value)
    print("true vars",true_var1_value,true_var2_value)
    return rand_prog_code, true_prog_code


@app.cell
def _():
    # i am making fause programs, which can have a number of sliders.
    # each slider has a value, and a range (min,max). the value is the default value, and the range is useful to set boundries for parameter updates
    # let's say the function that generates these programs is called generate_program.
    # each program could be a json file, where the program id and program code with placeholders for the parameters (parameters being valid ranges and values is defined)
    # generate_program takes in the program id, and optionally a set of values for each parameter. If the values are not defined, it will arbitrarly chooose a value between the range for each parameter. generate_program then returns the completed program, and a dictionary of parameter names and ranges. 

    return


@app.cell
def _(SAMPLE_RATE, args, fj, jax, rand_prog_code, true_prog_code):
    # L1_Spec , DTW_Onset, SIMSE_Spec, JTFS
    experiment = {
        "program_id": args.program_id,
        "loss": args.loss_fn,
        "lr": 0.045
    }

    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    true_instrument, true_instrument_jit, true_noise, true_instrument_params = fj.code_to_flax(true_prog_code, key)
    instrument, instrument_jit, noise, instrument_params = fj.code_to_flax(rand_prog_code, key)

    variable_names = [key.split("/")[1] for key in instrument_params["params"].keys()]

    print(instrument_params)
    return (
        experiment,
        instrument,
        instrument_jit,
        instrument_params,
        noise,
        true_instrument_params,
        true_noise,
        variable_names,
    )


@app.cell
def _(
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
    init_sound,true_params = instrument_jit(instrument_params, noise, SAMPLE_RATE)
    target_sound,_ = instrument_jit(true_instrument_params, true_noise, SAMPLE_RATE)
    fj.show_audio(init_sound)
    fj.show_audio(target_sound)
    return (target_sound,)


@app.cell
def _(
    SAMPLE_RATE,
    clip_spec,
    dm_pix,
    dtw_jax,
    experiment,
    instrument,
    instrument_jit,
    instrument_params,
    jax,
    jnp,
    kernel,
    naive_loss,
    noise,
    np,
    onset_1d,
    optax,
    scat_jax,
    spec_func,
    target_sound,
    train_state,
    variable_names,
):
    learning_rate = experiment["lr"]
    lfn = experiment["loss"]
    # Create Train state
    tx = optax.rmsprop(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )
    # loss fn shows the difference between the output of synth and a target_sound
    def loss_fn(params):
        pred = instrument_jit(params, noise, SAMPLE_RATE)[0]

        if lfn == 'L1_Spec':
            loss = naive_loss(spec_func(pred)[0], spec_func(target_sound))
        elif lfn == 'SIMSE_Spec':
            loss = dm_pix.simse(clip_spec(spec_func(target_sound)), clip_spec(spec_func(pred)))
        elif lfn == 'DTW_Onset':
            loss = dtw_jax(
                onset_1d(target_sound, kernel, spec_func),
                onset_1d(pred, kernel, spec_func)
            )
        elif lfn == 'JTFS':
            loss = naive_loss(scat_jax(target_sound), scat_jax(pred))
        else:
            raise ValueError("Invalid value for loss")
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
    real_params = {k: [] for k in variable_names}  # will record parameters while searching
    norm_params = {k: [] for k in variable_names}  # will record parameters while searching

    for n in range(50):
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
    return losses, real_params


@app.cell
def _():
    return


@app.cell
def _(true_instrument_params):
    true_instrument_params
    return


@app.cell
def _(losses, mo, plt, real_params, true_instrument_params):
    mo.output.clear()
    fig1, axs = plt.subplots(1, len(real_params) + 1, figsize=(12, 3))
    axs[0].set_xlabel("time (s)")
    axs[0].set_ylabel("loss", color="white")
    axs[0].plot(losses, color="white")

    c = 0
    colors = ["red", "green", "blue", "purple"]
    for pname2, pvalue in real_params.items():
        ax = axs[c + 1]
        ax.set_ylabel(pname2)  # we already handled the x-label with ax1
        ax.plot(real_params[pname2], color=colors[c])
        ax.axhline(true_instrument_params["params"]["_dawdreamer/%s" % pname2], linestyle="dashed", color=colors[c], label="true")
        ax.tick_params(axis="y")
        c += 1
    fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    fig1
    return


@app.cell
def _(mo):
    mo.md("We're dealing with a 2D loss function here, so lets isolate one of the parameters and plot the loss function as a function of the other parameter.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

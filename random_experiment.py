import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


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
    from helpers.experiment_scripts import append_to_json

    from kymatio.jax import Scattering1D
    import json
    import argparse

    import pickle
    import uuid


    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return (
        Path,
        SAMPLE_RATE,
        Scattering1D,
        append_to_json,
        argparse,
        copy,
        default_device,
        dm_pix,
        fj,
        functools,
        itertools,
        jax,
        jnp,
        json,
        length_seconds,
        librosa,
        loss_helpers,
        nn,
        np,
        optax,
        os,
        partial,
        pd,
        pickle,
        plt,
        softdtw_jax,
        train_state,
        unfreeze,
        uuid,
        wavfile,
    )


@app.cell
def __(argparse):
    # Parse known and unknown arguments
    # Create the parser
    parser = argparse.ArgumentParser(description='Process a loss function name.')

    # Add a string argument
    parser.add_argument('--loss_fn', type=str, help='the name of the loss function. One of:  L1_Spec , DTW_Onset, SIMSE_Spec, JTFS',default="L1_Spec")
    parser.add_argument('--learning_rate', type=float, help='learning rate',default=0.01)
    parser.add_argument('--program_id', type=int, choices=[0, 1, 2, 3], default = 0, help="The program ID to select (0, 1, 2, or 3)")
    args, unknown = parser.parse_known_args()

    # Parse the arguments
    # args = parser.parse_args()


    # Use the argument
    print(f'Loss function: {args.loss_fn}')
    return args, parser, unknown


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

    dtw_jax = softdtw_jax.SoftDTW(gamma=1)
    kernel = jnp.array(loss_helpers.gaussian_kernel1d(3, 0, 10))  # create a gaussian kernel (sigma,order,radius)

    J = 6  # higher creates smoother loss but more costly
    Q = 1
    scat_jax = Scattering1D(J, SAMPLE_RATE, Q)

    onset_1d = loss_helpers.onset_1d

    # for Multi_Spec loss. Generate spec functions for each NFFT value
    spec_funs = [loss_helpers.return_mel_spec(x, SAMPLE_RATE) for x in [512, 1024, 2048, 4096]]
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
        spec_funs,
    )


@app.cell
def __(args):
    from helpers.program_generators import choose_program
    import random

    if args.program_id == 0:
        var1_range = (50, 1000)
        var2_range = (1, 120)
        true_var1 = int(random.uniform(var1_range[0], var1_range[1]))
        true_var2 = int(random.uniform(var2_range[0], var2_range[1]))
    elif args.program_id == 1:
        var1_range = (30, 1000)
        var2_range = (30, 1000)
        true_var1 = int(random.uniform(var1_range[0], var1_range[1]))
        true_var2 = int(random.uniform(var2_range[0], var2_range[1]))
    elif args.program_id == 2:
        var1_range = (1, 20)
        var2_range = (10, 1000)
        true_var1 = int(random.uniform(var1_range[0], var1_range[1]))
        true_var2 = int(random.uniform(var2_range[0], var2_range[1]))
    elif args.program_id == 3:
        var1_range = (1, 20)
        var2_range = (10, 1000)
        true_var1 = int(random.uniform(var1_range[0], var1_range[1]))
        true_var2 = int(random.uniform(var2_range[0], var2_range[1]))


    rand_prog_code, var1_value, var2_value = choose_program(args.program_id, var1_range, var2_range)
    true_prog_code, true_var1_value, true_var2_value = choose_program(args.program_id,var1_range, var2_range,true_var1,true_var2)
    print("Program Code:\n", true_prog_code)
    print("init vars",var1_value,var2_value)
    print("true vars",true_var1_value,true_var2_value)
    return (
        choose_program,
        rand_prog_code,
        random,
        true_prog_code,
        true_var1,
        true_var1_value,
        true_var2,
        true_var2_value,
        var1_range,
        var1_value,
        var2_range,
        var2_value,
    )


@app.cell
def __(SAMPLE_RATE, args, fj, jax, rand_prog_code, true_prog_code):
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
        key,
        noise,
        true_instrument,
        true_instrument_jit,
        true_instrument_params,
        true_noise,
        variable_names,
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
    # Create Train state
    tx = optax.rmsprop(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=instrument.apply, params=instrument_params, tx=tx
    )
    lfn = experiment["loss"]

    # loss fn shows the difference between the output of synth and a target_sound
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

    for n in range(200):
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
        lfn,
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
def __(
    dtw_jax,
    experiment,
    kernel,
    loss_helpers,
    naive_loss,
    norm_params,
    onset_1d,
    pickle,
    program_id,
    scat_jax,
    sounds,
    spec_func,
    spec_funs,
    target_sound,
    true_instrument_params,
    uuid,
):
    # variables that need saving
    experiment["true_params"] = true_instrument_params
    experiment["norm_params"] = norm_params
    experiment["Multi_Spec"] = loss_helpers.loss_multi_spec(sounds[-1], target_sound,spec_funs)
    experiment["L1_Spec"] = naive_loss(spec_func(sounds[-1]), spec_func(target_sound))
    experiment["DTW_Onset"] = dtw_jax(onset_1d(target_sound, kernel, spec_func), onset_1d(sounds[-1], kernel, spec_func))
    experiment["JTFS"] = naive_loss(scat_jax(target_sound), scat_jax(sounds[-1]))
    experiment["target_sound"] = target_sound
    experiment["output_sound"] = sounds[-1]

    # Generate a random file name
    file_name = f"./results/%s_%s_%s.pkl"%(experiment["loss"],program_id,uuid.uuid4())

    # Save the dictionary with the random file name
    with open(file_name, "wb") as file:
        pickle.dump(experiment, file)

    print(f"File saved as: {file_name}")
    return file, file_name


@app.cell
def __(file_name):
    file_name
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

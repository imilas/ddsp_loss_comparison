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
    from kymatio.jax import Scattering1D
    import json
    import argparse

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return (
        Path,
        SAMPLE_RATE,
        Scattering1D,
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
        plt,
        softdtw_jax,
        train_state,
        unfreeze,
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
def __(args):
    from helpers.program_generators import choose_program

    var1_range = (50, 1000)
    var2_range = (1, 120)
    true_var1 = sum(var1_range)//2
    true_var2 = sum(var2_range)//2

        # prog_code, var1_value, var2_value, var_names = choose_program(args.program_id, var1_range, var2_range)

    rand_prog_code, var1_value, var2_value = choose_program(args.program_id, var1_range, var2_range)
    true_prog_code, true_var1_value, true_var2_value = choose_program(args.program_id,var1_range, var2_range,true_var1,true_var2)
    print("Program 0 Code:\n", true_prog_code)
    print("init vars",var1_value,var2_value)
    print("true vars",true_var1_value,true_var2_value)
    return (
        choose_program,
        rand_prog_code,
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
def __(norm_params):
    norm_params
    return


@app.cell
def __(grad_fn, instrument_params, np, pd):
    def make_programs_df(true_params, granularity):
        # make programs that cover the grid, given the parameters dict
        meshgrid = np.meshgrid(*[np.linspace(-0.95, 1, granularity, endpoint=False) for i in range(len(true_params.keys()))])
        program_params = np.stack([d.flatten() for d in meshgrid], axis=-1)
        programs_df = pd.DataFrame(program_params, columns=true_params.keys())
        return programs_df


    granularity = 5
    programs_df = make_programs_df(instrument_params["params"], granularity)
    grad_loss_dict = [grad_fn({"params": programs_df.loc[i].to_dict()}) for i in range(len(programs_df))]
    return grad_loss_dict, granularity, make_programs_df, programs_df


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

    plt.imshow(data, cmap="Blues", extent=(-1, 1, -1, 1), origin="lower")
    # Add a color bar on the right
    # plt.colorbar(label='loss',orientation="vertical",aspect=5)

    grad_dir = np.array([list(x[1]["params"].values()) for x in grad_loss_dict])
    # grad_dir = np.where(grad_dir>0,1,-1)
    plt.quiver(*programs_df.T.to_numpy(), *-grad_dir.T)
    plt.scatter(*true_instrument_params["params"].values(), edgecolors="black",linewidth=1,color="#00FF00", marker="*", s=[550],label="Target")
    plt.scatter(*instrument_params["params"].values(), color="#00FF00", marker="o", s=[250],edgecolors="black",label="Init. Params.")

    # path 
    simple_norm_params = {k: v[::15] for k, v in norm_params.items()}
    plt.scatter(*list(simple_norm_params.values()),color="#FF0000",alpha=1,s=[150],marker=".",edgecolors="black",linewidths=1,label="Update Path")
    plt.plot(*list(simple_norm_params.values()),color="#FF0000",alpha=1,marker=".")

    plt.xlim(-1, 1)   # Limit x-axis from 2 to 8
    plt.ylim(-1, 1)  # Limit y-axis from -0.5 to 0.

    plt.tight_layout()
    plt.xlabel(programs_df.columns[0].split("/")[-1])
    plt.ylabel(programs_df.columns[1].split("/")[-1])

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1),markerscale=0.6)
    # plt.tight_layout()
    # plt.subplots_adjust(right=1, top=0.95, bottom=0, left=0.0)

    plt.show()
    # plt.savefig("./plots/p%d_%s.png"%(experiment["program_id"],experiment["loss"]),bbox_inches='tight', pad_inches=0, transparent=True)
    return data, grad_dir, simple_norm_params


@app.cell
def __(experiment, norm_params, true_instrument_params):
    # variables that need saving
    experiment["true_params"] = true_instrument_params
    experiment["norm_params"] = norm_params
    experiment
    return


@app.cell
def __(experiment, json, np, os):

    def convert_to_serializable(obj):
        """
        Recursively convert non-serializable objects (like NumPy arrays) to serializable types.
        """
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        elif hasattr(obj, 'tolist'):  # If the object has a 'tolist' method (like some tensor types)
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj  # Directly serializable types
        else:
            return str(obj)  # Convert unknown objects to their string representation

    def append_to_json(file_path, new_data):
        """
        Appends a dictionary to a JSON file. If the file doesn't exist, it creates a new one.
        
        Args:
            file_path (str): Path to the JSON file.
            new_data (dict): Dictionary to append.
        """
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the existing content
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    # If the file is empty or contains invalid JSON, initialize as an empty list
                    data = []
        else:
            # If the file does not exist, initialize as an empty list
            data = []

        # Convert new_data and its contents to serializable types
        new_data = convert_to_serializable(new_data)

        # Append the new data (ensure the file is a list)
        if isinstance(data, list):
            data.append(new_data)
        else:
            raise ValueError("Expected the JSON file to contain a list")

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    # Specify the path to the JSON file
    json_file_path = 'results/experiments.json'

    # Example experiment dictionary (modify it based on your actual data)
    # experiment = {
    #     "program_id": 0,
    #     "loss": "L1_Spec",
    #     "lr": 0.045,
    #     "params": {
    #         "_dawdreamer/hp_cut": np.array([-0.008403360843658447]),
    #         "_dawdreamer/lp_cut": np.array([0.0])
    #     }
    # }

    # Call the function to append to the JSON file
    append_to_json(json_file_path, experiment)

    # Output the experiment for verification
    experiment

    return append_to_json, convert_to_serializable, json_file_path


@app.cell
def __(json, os):

    def load_json(file_path):
        """
        Loads a JSON file and returns its content.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            dict or list: The content of the JSON file.
            None: If the file does not exist or is empty.
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    return data
                except json.JSONDecodeError:
                    print("Error: The file contains invalid JSON.")
                    return None
        else:
            print(f"Error: The file '{file_path}' does not exist.")
            return None



    load_json("results/experiments.json")
    return load_json,


if __name__ == "__main__":
    app.run()

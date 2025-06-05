import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    _parentdir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(_parentdir))


    import sys
    from pathlib import Path
    import argparse
    import copy
    from functools import partial
    import jax
    import jax.numpy as jnp
    from jax import random as jrandom
    import optax
    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state

    import marimo as mo

    from helper_funcs import faust_to_jax as fj
    from helper_funcs import program_generators as pg
    from helper_funcs import experiment_setup as setup

    import matplotlib.pyplot as plt
    import numpy as np
    import random

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    fj.SAMPLE_RATE = SAMPLE_RATE

    # experiment setup
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

    # L1_Spec , DTW_Onset, SIMSE_Spec, JTFS
    experiment = {
        "program_id": args.program_id,
        "loss": args.loss_fn,
        "lr": 0.045
    }

    return (
        SAMPLE_RATE,
        args,
        clip_spec,
        copy,
        dtw_jax,
        experiment,
        fj,
        jax,
        jnp,
        kernel,
        mo,
        naive_loss,
        np,
        onset_1d,
        optax,
        pg,
        plt,
        scat_jax,
        spec_func,
        train_state,
    )


@app.cell
def _(SAMPLE_RATE, fj, jax, mo, pg):
    target_prog_code, target_var1, target_var2 = pg.generate_program_3((1, 20),(1,250))
    imitator_prog_code, imitator_va1, imitator_var2 = pg.generate_program_3((1, 20),(1000, 5000))

    # imitator_prog_code, imitator_va1, imitator_var2 = pg.generate_program_2((0.1, 1),  (1, 20))


    print(target_prog_code, target_var1, target_var2)
    print(imitator_prog_code, imitator_va1, imitator_var2)

    key = jax.random.PRNGKey(10)

    target_instrument, target_instrument_jit, target_noise, target_instrument_params = fj.code_to_flax(target_prog_code, key)
    imitator_instrument, imitator_instrument_jit, imitator_noise, imitator_instrument_params = fj.code_to_flax(imitator_prog_code, key)

    mo.output.clear()
    target_sound = target_instrument_jit(target_instrument_params, target_noise, SAMPLE_RATE)[0]
    imitator_sound = imitator_instrument_jit(imitator_instrument_params, imitator_noise, SAMPLE_RATE)[0]
    fj.show_audio(target_sound)
    fj.show_audio(imitator_sound)
    return (
        imitator_instrument,
        imitator_instrument_jit,
        imitator_instrument_params,
        imitator_noise,
        target_instrument_params,
        target_sound,
    )


@app.cell
def _(
    SAMPLE_RATE,
    args,
    clip_spec,
    dm_pix,
    dtw_jax,
    imitator_instrument_jit,
    imitator_noise,
    jax,
    kernel,
    loss_multi_spec,
    naive_loss,
    onset_1d,
    scat_jax,
    spec_func,
    target_sound,
):
    args.loss_fn = "DTW_Onset"
    lfn = args.loss_fn
    def loss_fn(params):
        pred = imitator_instrument_jit(params, imitator_noise, SAMPLE_RATE)[0]
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
    return (grad_fn,)


@app.cell
def _(
    SAMPLE_RATE,
    experiment,
    grad_fn,
    imitator_instrument,
    imitator_instrument_jit,
    imitator_instrument_params,
    imitator_noise,
    jax,
    jnp,
    np,
    optax,
    train_state,
):
    learning_rate = experiment["lr"]
    # Create Train state
    tx = optax.rmsprop(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=imitator_instrument.apply, params=imitator_instrument_params, tx=tx
    )

    # Clip gradients function
    def clip_grads(grads, clip_norm):
        total_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(grads)))
        scale = clip_norm / jnp.maximum(total_norm, clip_norm)
        return jax.tree_util.tree_map(lambda g: g * scale, grads)


    @jax.jit
    def train_step(state):
        """Train for a single step."""
        # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        grads = clip_grads(grads, clip_norm=1.0)
        state = state.apply_gradients(grads=grads)
        return state, loss

    variable_names = [key.split("/")[1] for key in imitator_instrument_params["params"].keys()]
    losses = []
    sounds = []
    real_params = {k: [] for k in variable_names}  # will record parameters while searching
    norm_params = {k: [] for k in variable_names}  # will record parameters while searching

    for n in range(200):
        state, loss = train_step(state)
        if n % 1 == 0:
            audio, mod_vars = imitator_instrument_jit(state.params, imitator_noise, SAMPLE_RATE)
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
    return losses, real_params, sounds


@app.cell
def _(losses, plt, real_params):
    def plot_params_and_loss(real_params, losses):
        variable_names = list(real_params.keys())
        num_vars = len(variable_names)
        total_plots = num_vars + 1  # Add one for loss

        fig, axs = plt.subplots(1, total_plots, figsize=(5 * total_plots, 4))

        for i, var in enumerate(variable_names):
            axs[i].plot(real_params[var])
            axs[i].set_title(f"{var}")
            axs[i].set_xlabel("Step")
            axs[i].set_ylabel("Value")

        axs[-1].plot(losses)
        axs[-1].set_title("Loss")
        axs[-1].set_xlabel("Step")
        axs[-1].set_ylabel("Loss")

        plt.tight_layout()
        plt.show()

    # Example call:
    plot_params_and_loss(real_params, losses)
    return


@app.cell
def _(fj, mo, sounds, target_sound):
    mo.output.clear()
    fj.show_audio(target_sound)
    fj.show_audio(sounds[-1])
    return


@app.cell
def _(copy, jnp, target_instrument_params):
    def fill_template(template, pkey, fill_values):
        template = template.copy()
        """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we assign to the parameter
        """
        for i, k in enumerate(pkey):
            template["params"][k] = fill_values[i]
        return template
    
    target_param = list(target_instrument_params["params"].keys())[0]
    param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 300, endpoint=False))
    programs = [
        fill_template(copy.deepcopy(target_instrument_params), [target_param], [x])
        for x in param_linspace
    ]
    return


@app.cell
def _(np):
    import itertools

    def generate_param_grid(param_ranges, granularities):
        """
        param_ranges: dict like {"param_name": (min, max)}
        granularities: dict like {"param_name": N}
    
        Returns: list of dictionaries, each representing one param combination
        """
        assert param_ranges.keys() == granularities.keys()
    
        # Create list of values for each param
        param_values = {
            k: np.linspace(param_ranges[k][0], param_ranges[k][1], granularities[k])
            for k in param_ranges
        }

        # Get Cartesian product of all parameter values
        all_combinations = list(itertools.product(*param_values.values()))
    
        # Convert tuples to dictionaries
        param_names = list(param_values.keys())
        grid = [dict(zip(param_names, combo)) for combo in all_combinations]
        return grid
    param_ranges = {
        "_dawdreamer/amp": (-1, 1),
        "_dawdreamer/carrier": (-1, 1),
    }
    granularities = {
        "_dawdreamer/amp": 20,
        "_dawdreamer/carrier": 20,
    }

    grid = generate_param_grid(param_ranges, granularities)
    print(f"Generated {len(grid)} combinations.")
    print(grid)  # Example output
    return granularities, param_ranges


@app.cell
def _():
    return


@app.cell
def _(grad_fn, granularities, np, param_ranges):
    from matplotlib import cm
    # Create value grids
    amp_vals = np.linspace(*param_ranges["_dawdreamer/amp"], granularities["_dawdreamer/amp"])
    carrier_vals = np.linspace(*param_ranges["_dawdreamer/carrier"], granularities["_dawdreamer/carrier"])
    loss_grid = np.zeros((len(carrier_vals), len(amp_vals)))

    # Evaluate loss for each parameter combination
    for i, c in enumerate(carrier_vals):
        for j, a in enumerate(amp_vals):
            program = {
                "params": {
                    "_dawdreamer/amp": a,
                    "_dawdreamer/carrier": c,
                }
            }
            (l, _), _ = grad_fn(program)
            loss_grid[i, j] = l


    return amp_vals, carrier_vals, cm, loss_grid


@app.cell
def _():
    # target_amp = target_instrument_params["params"]["_dawdreamer/amp"]
    # target_carrier = target_instrument_params["params"]["_dawdreamer/carrier"]

    # # Plot 2D heatmap
    # A, C = np.meshgrid(amp_vals, carrier_vals)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # cols = ax.pcolormesh(A, C, loss_grid, shading='auto', cmap=cm.viridis)
    # plt.colorbar(cols, ax=ax, label='Loss')

    # # Add target point and origin
    # ax.plot(target_amp, target_carrier, 'ro', markersize=6, label='Target Params')
    # ax.plot(0, 0, 'bo', markersize=6, label='Origin (0,0)')

    # # Labels and formatting
    # ax.set_xlabel("amp")
    # ax.set_ylabel("carrier")
    # ax.set_title("Loss heatmap over parameter grid")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def _(
    amp_vals,
    carrier_vals,
    cm,
    loss_grid,
    np,
    plt,
    target_instrument_params,
):
    # Create meshgrid for plotting
    A, C = np.meshgrid(amp_vals, carrier_vals)

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, C, loss_grid, cmap=cm.viridis)

    # Coordinates for the vertical line
    x0 = 0
    y0 = 0
    target_amp = target_instrument_params["params"]["_dawdreamer/amp"]
    target_carrier = target_instrument_params["params"]["_dawdreamer/carrier"]


    # Draw vertical line from z=0 to loss_val
    ax.plot(
        [target_amp, target_amp],           # X: constant
        [target_carrier, target_carrier],           # Y: constant
        [0, np.max(loss_grid)],      # Z: from base to top
        color='red', linewidth=2, label='(0, 0) Reference'
    )

    ax.set_xlabel("amp")
    ax.set_ylabel("carrier")
    ax.set_zlabel("loss")
    ax.set_title("Loss surface over parameter grid")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()

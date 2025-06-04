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
    return SAMPLE_RATE, fj, jax, mo, pg


@app.cell
def _(pg):
    # target code is program 3
    target_range_1 = (1, 20)
    target_range_2= (10, 1000)
    target_prog_code, target_var1, target_var2 = pg.generate_program_3(target_range_1, target_range_2)


    imitator_range_1 = (0.1, 1)
    imitator_range_2 = (1, 20)
    imitator_prog_code, imitator_va1, imitator_var2 = pg.generate_program_2(imitator_range_1, imitator_range_2)


    print(target_prog_code, target_var1, target_var2)
    print(imitator_prog_code, imitator_va1, imitator_var2)
    return imitator_prog_code, target_prog_code


@app.cell
def _(fj, imitator_prog_code, jax, target_prog_code):
    key = jax.random.PRNGKey(10)

    target_instrument, target_instrument_jit, target_noise, target_instrument_params = fj.code_to_flax(target_prog_code, key)
    imitator_instrument, imitator_instrument_jit, imitator_noise, imitator_instrument_params = fj.code_to_flax(imitator_prog_code, key)

    return (
        imitator_instrument_jit,
        imitator_instrument_params,
        imitator_noise,
        target_instrument_jit,
        target_instrument_params,
        target_noise,
    )


@app.cell
def _(
    SAMPLE_RATE,
    fj,
    imitator_instrument_jit,
    imitator_instrument_params,
    imitator_noise,
    mo,
    target_instrument_jit,
    target_instrument_params,
    target_noise,
):
    mo.output.clear()
    target_sound = target_instrument_jit(target_instrument_params, target_noise, SAMPLE_RATE)[0]
    imitator_sound = imitator_instrument_jit(imitator_instrument_params, imitator_noise, SAMPLE_RATE)[0]
    fj.show_audio(target_sound)
    fj.show_audio(imitator_sound)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

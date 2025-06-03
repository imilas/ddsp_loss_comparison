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

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

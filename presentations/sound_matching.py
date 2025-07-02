import marimo

__generated_with = "0.13.6"
app = marimo.App(
    width="full",
    layout_file="layouts/sound_matching.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    _parentdir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(_parentdir))

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
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt

    # from audax.core import functional
    import copy
    import dm_pix

    from helper_funcs import faust_to_jax as fj
    from helper_funcs import loss_helpers
    from helper_funcs import softdtw_jax
    from kymatio.jax import Scattering1D


    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be
    return (mo,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10)
    mo.md(f"Choose a value: {slider}")
    return


@app.cell
def _(mo):
    mo.md("""here is an image""")
    mo.md("![/plots/p0_MSS.png]")


    return


@app.cell
def _(mo):
    _src = (
        "plots/p0_DTW_Onset.png"
    )
    mo.image(src=_src,  rounded=True)
    return


if __name__ == "__main__":
    app.run()

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
    return (
        SAMPLE_RATE,
        argparse,
        copy,
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
def _(SAMPLE_RATE, fj, jax, pg):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    # faust_code,_ = pg.generate_1_1d([100,20000])
    faust_code, _ = pg.generate_2_1d([0.1,20])
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
    lpg = [grad_fn(p) for p in programs] # (loss,preduct), params for each program
    g = [list(p[1]["params"].values())[0] for p in lpg] # get grads for each program
    return g, lpg


@app.cell
def _(DSP_params, g, lpg, np, param_linspace, plt):


    FIGSIZE = (5, 3)  # Inches
    DPI = 300
    SAVE_KWARGS = dict(dpi=DPI, bbox_inches='tight', pad_inches=0, facecolor='white')

    fig, ax1 = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    # Main plot
    ax1.plot(param_linspace, [p[0][0] for p in lpg], color='blue')
    ax1.set_ylabel("Loss", color='blue')

    # Second y-axis
    ax2 = ax1.twinx()
    smoothed_grad = np.convolve(g, np.ones(40)/40, mode='same')
    ax2.plot(param_linspace, smoothed_grad, color='green')
    ax2.set_ylabel("Gradients", color='green')

    # Red vertical line
    correct_param = list(DSP_params["params"].values())[0]
    ax1.axvline(x=correct_param, color='red', linestyle='--', linewidth=2, label="Correct Param")

    fig.suptitle("DTW Landscape Example")
    fig.legend(bbox_to_anchor=(0.75, 0.9))

    fig.savefig("./plots/DTW_loss_landscape.png", **SAVE_KWARGS)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

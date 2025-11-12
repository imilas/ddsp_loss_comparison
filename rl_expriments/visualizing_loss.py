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
    import dm_pix
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
        dm_pix,
        fj,
        jax,
        jnp,
        length_seconds,
        mo,
        np,
        partial,
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


app._unparsable_cell(
    r"""
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code,_ = pg.generate_0_1d([100,20000],lp_cut=15000)
    faust_code, _ = pg.generate_2_1d([0.1,20])-
    print(faust_code)
    """,
    name="_"
)


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
    param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 100, endpoint=False))
    programs = [
        fill_template(copy.deepcopy(DSP_params), [target_param], [x])
        for x in param_linspace
    ]

    s,_ = instrument_jit(programs[25], noise, SAMPLE_RATE)
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
    np,
    onset_1d,
    plt,
    programs,
    scat_jax,
    spec_func,
    target_sound,
):
    # --- LOSS LANDSCAPES FOR MULTIPLE LOSSES (normalize to 0–1) ---
    # Force white theme
    plt.style.use("default")
    loss_names = ['L1_Spec',"SIMSE_Spec","DTW_Onset","JTFS"]  # adjust as needed

    def make_grad_fn(loss_name):
        # Localized wrapper to avoid changing your existing loss_fn/grad_fn globals
        def _loss_fn(params):
            pred = instrument_jit(params, noise, SAMPLE_RATE)[0]
            if loss_name == 'L1_Spec':
                loss = naive_loss(spec_func(pred)[0], spec_func(target_sound))
            elif loss_name == 'SIMSE_Spec':
                loss = dm_pix.simse(clip_spec(spec_func(target_sound)), clip_spec(spec_func(pred)))
            elif loss_name == 'DTW_Onset':
                loss = dtw_jax(onset_1d(target_sound, kernel, spec_func), onset_1d(pred, kernel, spec_func))
            elif loss_name == 'JTFS':
                loss = naive_loss(scat_jax(target_sound), scat_jax(pred)[0])
            elif loss_name == 'Multi_Spec':
                loss = loss_multi_spec(target_sound, pred)
            else:
                raise ValueError(f"Invalid loss_name: {loss_name}")
            return loss, pred
        return jax.jit(jax.value_and_grad(_loss_fn, has_aux=True))

    # Compute loss/grad sweeps for each loss function across your existing `programs`
    loss_curves_norm = {}  # name -> np.array normalized to [0, 1]
    grad_curves_norm = {}  # name -> np.array normalized to [-1, 1]

    for name in loss_names:
        gf_local = make_grad_fn(name)
        sweep_outputs = [gf_local(p) for p in programs]  # [( (loss, pred), grads ), ...]
        # Extract raw losses (as scalar) and raw grads for the first parameter
        loss_values_raw = np.array([float(out[0][0]) for out in sweep_outputs])
        grad_values_raw = np.array([float(list(out[1]["params"].values())[0]) for out in sweep_outputs])

        # Normalize losses to [0, 1] per-curve
        l_min = np.min(loss_values_raw)
        l_max = np.max(loss_values_raw)
        loss_norm = (loss_values_raw - l_min) / (l_max - l_min + 1e-12)

        # Optionally smooth gradients (comment out if not desired)
        # grad_values_raw = np.convolve(grad_values_raw, np.ones(40)/40, mode='same')

        # Normalize gradients to [-1, 1] per-curve
        g_abs_max = np.max(np.abs(grad_values_raw)) + 1e-12
        grad_norm = grad_values_raw / g_abs_max

        loss_curves_norm[name] = loss_norm
        grad_curves_norm[name] = grad_norm

    FIGSIZE = (6, 4)
    DPI = 300

    return FIGSIZE, grad_curves_norm, loss_curves_norm, loss_names


@app.cell
def _(DSP_params, FIGSIZE, loss_curves_norm, loss_names, param_linspace, plt):
    fig, ax1 = plt.subplots(figsize=FIGSIZE, constrained_layout=True, )

    # # Red vertical line
    correct_param = list(DSP_params["params"].values())[0]
    ax1.axvline(x=correct_param, color='red', linestyle='--', linewidth=2, label="Correct Param")
    # Plot normalized loss landscapes (0–1) on a single axis
    fig_loss, ax_loss = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    for name0 in loss_names:
        ax_loss.plot(param_linspace, loss_curves_norm[name0], label=name0)
    ax_loss.set_xlabel("Parameter")
    ax_loss.set_ylabel("Normalized Loss (0–1)")
    ax_loss.axvline(x=correct_param, color='red', linestyle='--', linewidth=2, label="Correct Param")
    # ax_loss.set_title("Normalized Loss Landscapes")
    ax_loss.legend(bbox_to_anchor=(0., 0.345),loc="upper left")
    fig_loss.savefig(f"./plots/comparing_lanscapes_1_1d_normalized.png")
    return (correct_param,)


@app.cell
def _(
    FIGSIZE,
    correct_param,
    grad_curves_norm,
    loss_names,
    param_linspace,
    plt,
):
    # --- GRADIENT PLOTS (normalize to −1..1) ---

    fig_grad, ax_grad = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    for name1 in loss_names:
        ax_grad.plot(param_linspace, grad_curves_norm[name1], label=name1)
    ax_grad.set_xlabel("Parameter")
    ax_grad.set_ylabel("Normalized Gradient (−1 to 1)")
    ax_grad.axhline(0.0, color='k', linewidth=0.8, alpha=0.6)
    ax_grad.axvline(x=correct_param, color='red', linestyle='--', linewidth=2, label="Correct Param")
    # ax_grad.set_title("Normalized Gradients")
    ax_grad.legend(bbox_to_anchor=(0., 0.365),loc="upper left")
    fig_grad
    # fig_grad.savefig("./plots/loss_gradients_normalized.png", **SAVE_KWARGS)
    return


if __name__ == "__main__":
    app.run()

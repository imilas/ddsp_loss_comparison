import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import functools
    from functools import partial
    import itertools
    from pathlib import Path
    import os
    import jax
    import jax.numpy as jnp

    from flax import linen as nn
    import optax

    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt

    from helpers import faust_to_jax as fj
    from audax.core import functional
    import copy
    from helpers import ts_comparisions as ts_comparisons
    import dtw

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)

    SAMPLE_RATE = 44100
    length_seconds = 1  # how long should samples be


    naive_loss = lambda x, y: np.abs(x - y).mean()
    cosine_distance = lambda x, y: np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y)
    )
    return (
        SAMPLE_RATE,
        copy,
        dtw,
        fj,
        jax,
        jnp,
        length_seconds,
        librosa,
        mo,
        naive_loss,
        np,
        partial,
        plt,
        ts_comparisons,
    )


@app.cell
def _(SAMPLE_RATE, fj, jax):
    fj.SAMPLE_RATE = SAMPLE_RATE
    key = jax.random.PRNGKey(10)

    faust_code_3 = """
    import("stdfaust.lib");
    f = hslider("freq",1.,0,5,0.1);
    peak_f = hslider("peak_f",40,40,200,1);
    process = os.saw4(os.osc(f)*peak_f);
    """
    return faust_code_3, key


@app.cell
def _(SAMPLE_RATE, faust_code_3, fj, jax, key, length_seconds, partial):
    DSP = fj.faust2jax(faust_code_3)
    DSP = DSP(SAMPLE_RATE)
    DSP_jit = jax.jit(partial(DSP.apply,mutable="intermediates"), static_argnums=[2])
    noise = jax.random.uniform(
        jax.random.PRNGKey(10),
        [DSP.getNumInputs(), SAMPLE_RATE * length_seconds],
        minval=-1,
        maxval=1,
    )
    DSP_params = DSP.init(key, noise, SAMPLE_RATE)
    return DSP_jit, DSP_params, noise


@app.cell
def _(DSP_params):
    DSP_params
    return


@app.cell
def _(faust_code_3, fj, key, length_seconds, mo):
    mo.output.clear()
    target, _ = fj.process_noise_in_faust(
        faust_code_3, key, length_seconds=length_seconds
    )
    fj.show_audio(target)
    return (target,)


@app.cell
def _(DSP_jit, DSP_params, SAMPLE_RATE, copy, jnp, noise):
    def fill_template(template, pkey, fill_values):
        template = template.copy()
        """template is the model parameter, pkey is the parameter we want to change, and fill_value is the value we assign to the parameter
        """
        for i, k in enumerate(pkey):
            template["params"][k] = fill_values[i]
        return template


    target_param = "_dawdreamer/freq"
    param_linspace = jnp.array(jnp.linspace(-0.99, 1.0, 300, endpoint=False))
    programs = [
        fill_template(copy.deepcopy(DSP_params), [target_param], [x])
        for x in param_linspace
    ]

    outputs = [DSP_jit(p, noise, SAMPLE_RATE)[0] for p in programs]
    return outputs, param_linspace, target_param


@app.cell
def _(fj, mo, outputs):
    mo.output.clear()
    fj.show_audio(outputs[0])
    return


@app.cell
def _(SAMPLE_RATE, librosa, np, outputs, target):
    output_onsets = [
        librosa.onset.onset_strength_multi(
            y=np.array(y), sr=SAMPLE_RATE, channels=[0, 16, 64, 96, 128]
        )
        for y in outputs
    ]
    target_onset = librosa.onset.onset_strength_multi(
        y=np.array(target), sr=SAMPLE_RATE, channels=[0, 16, 64, 96, 128]
    )
    return output_onsets, target_onset


@app.cell
def _(
    DSP_params,
    mo,
    np,
    output_onsets,
    param_linspace,
    plt,
    target_onset,
    target_param,
    ts_comparisons,
):
    mo.output.clear()
    # we calculate the onsets then use cbd loss

    cbd = ts_comparisons.CompressionBasedDissimilarity()

    cbd_loss = [
        cbd.calculate(
            np.array(target_onset[0]).sum(axis=0), np.array(x).sum(axis=0)
        )
        for x in output_onsets
    ]

    plt.plot(param_linspace, cbd_loss)
    plt.axvline(
        DSP_params["params"][target_param],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("cbd loss using onsets")
    return


@app.cell
def _(
    DSP_params,
    dtw,
    mo,
    np,
    output_onsets,
    param_linspace,
    plt,
    target_onset,
    target_param,
):
    mo.output.clear()


    def dtw_loss(x1, x2):

        query = np.array(x1).T
        template = np.array(x2).T
        # query = np.array(x1).sum(axis=0)
        # template = np.array(x2).sum(axis=0)
        alignment = dtw.dtw(
            query,
            template,
            keep_internals=True,
            step_pattern=dtw.rabinerJuangStepPattern(6, "c"),
        )
        return alignment.normalizedDistance


    dtw_losses = [dtw_loss(target_onset[0], x) for x in output_onsets]

    plt.plot(param_linspace, dtw_losses)
    plt.axvline(
        DSP_params["params"][target_param],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("dtw loss using onsets")
    return


@app.cell
def _():
    # mo.output.clear()
    # from kymatio.torch import Scattering1D  # faster than the numpy, jax very slow
    # import torch

    # J = 8  # higher creates smoother loss but more costly
    # Q = 16

    # scat_torch = Scattering1D(J, SAMPLE_RATE, Q)
    # meta = scat_torch.meta()
    # order0 = np.where(meta["order"] == 0)
    # order1 = np.where(meta["order"] == 1)
    # order2 = np.where(meta["order"] == 2)


    # log_eps = 1e-6
    # target_scatter = scat_torch(torch.asarray(target[0]))
    # # target_scatter = target_scatter[1:, :]
    # target_scatter = torch.log(torch.abs(target_scatter) + log_eps)
    # target_scatter = torch.mean(target_scatter, dim=-1)

    # Sx_all = scat_torch.forward(torch.asarray(np.asarray(outputs)))
    # # Sx_all_0 = Sx_all[:, 1:, :]
    # Sx_all = torch.log(torch.abs(Sx_all) + log_eps)
    # Sx_all = torch.mean(Sx_all, dim=-1)


    # losses_scatter = [naive_loss(x, target_scatter) for x in Sx_all]
    # plt.plot(param_linspace, losses_scatter)
    # plt.axvline(
    #     DSP_params["params"][target_param],
    #     color="#FF000055",
    #     linestyle="dashed",
    #     label="correct param",
    # )
    # plt.legend()
    # plt.title("wavelet scatter loss")
    return


@app.cell
def _(SAMPLE_RATE, mo, np, plt, target):
    mo.output.clear()
    from kymatio.jax import Scattering1D

    J = 4 # higher creates smoother loss but more costly
    Q = 1

    scat_jax = Scattering1D(J, SAMPLE_RATE, Q)
    # scat_jit = jax.jit(scat_jax)
    meta = scat_jax.meta()
    order0 = np.where(meta["order"] == 0)
    order1 = np.where(meta["order"] == 1)
    order2 = np.where(meta["order"] == 2)
    log_eps = 1e-6

    target_scatter = scat_jax(target)[0]
    plt.plot(target_scatter.sum(axis=0))
    # # target_scatter = target_scatter[1:, :]
    # target_scatter = torch.log(torch.abs(target_scatter) + log_eps)
    # target_scatter = torch.mean(target_scatter, dim=-1)
    return scat_jax, target_scatter


@app.cell
def _(outputs, scat_jax):
    outputs_scatter = [scat_jax(x[0]) for x in outputs]
    # outputs_scatter = [scat_jit(x[0]) for x in outputs[0:10]]
    return (outputs_scatter,)


@app.cell
def _(
    DSP_params,
    naive_loss,
    outputs_scatter,
    param_linspace,
    plt,
    target_param,
    target_scatter,
):
    losses_scatter = [naive_loss(x, target_scatter) for x in outputs_scatter]
    plt.plot(param_linspace, losses_scatter)
    plt.axvline(
        DSP_params["params"][target_param],
        color="#FF000055",
        linestyle="dashed",
        label="correct param",
    )
    plt.legend()
    plt.title("wavelet scatter loss")
    return


@app.cell
def _():
    # what to do?
    # - 2D loss function might be smoother
    # - show effectiveness on out of synth sounds
    #     - with sgd
    #     - with qdx
    # - out of synth situations:
    # - this can be different synth or organic
    # - compare dtw and scattering transforms
    return


@app.cell
def _():
    # NFFTs = [256,512,2048,4096] 

    # def return_mel_spec(NFFT):
    #     WIN_LEN = 400
    #     HOP_LEN = 20
    #     window = jnp.hanning(WIN_LEN)
    #     spec_func = partial(functional.spectrogram, pad=0, window=window, n_fft=NFFT,
    #                        hop_length=HOP_LEN, win_length=WIN_LEN, power=1,
    #                        normalized=False, center=False, onesided=True)
    #     fb = functional.melscale_fbanks(n_freqs=(NFFT//2)+1, n_mels=64,
    #                              sample_rate=SAMPLE_RATE, f_min=60., f_max=SAMPLE_RATE//2)
    #     mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)

    #     jax_spec = jax.jit(spec_func)
    #     mel_spec = jax.jit(mel_spec_func) 
    #     return mel_spec,jax_spec 

    # spec_funs = [return_mel_spec(x) for x in NFFTs]
    # spectrogram = spec_funs [-1][0](spec_funs[-1][1]((target)))
    # librosa.display.specshow(spectrogram[-1].T)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

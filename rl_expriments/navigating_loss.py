import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    import argparse
    import copy

    import jax
    import jax.numpy as jnp
    from jax import random as jrandom

    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state
    from flax.core.frozen_dict import unfreeze
    import optax

    import marimo as mo

    from helper_funcs import faust_to_jax as fj
    from helper_funcs import program_generators as pg
    from helper_funcs import experiment_setup as setup


    import matplotlib.pyplot as plt
    import numpy as np

    # Add parent directory to path for imports
    _parentdir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(_parentdir))

    # --- Config ---
    SAMPLE_RATE = 44100
    length_seconds = 1
    jax.config.update("jax_platform_name", "cpu")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Choose a loss function and program.')
    parser.add_argument('--loss_fn', type=str, default="L1_Spec")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--program_id', type=int, choices=[0, 1, 2, 3], default=0)
    args, _ = parser.parse_known_args()

    spec_func = setup.spec_func
    clip_spec = setup.clip_spec
    naive_loss = setup.naive_loss
    dtw_jax = setup.dtw_jax
    scat_jax = setup.scat_jax
    kernel = setup.kernel
    onset_1d = setup.onset_1d
    return (
        SAMPLE_RATE,
        clip_spec,
        dtw_jax,
        fj,
        jax,
        jnp,
        jrandom,
        kernel,
        length_seconds,
        mo,
        naive_loss,
        onset_1d,
        pg,
        plt,
        scat_jax,
        spec_func,
    )


@app.cell
def _(SAMPLE_RATE, fj, jax, jnp, jrandom, length_seconds, mo, pg):
    # --- Set Sample Rate in faust_to_jax ---
    fj.SAMPLE_RATE = SAMPLE_RATE

    # --- Generate Faust Program ---
    faust_code, _ = pg.generate_2_1d([0.1,20])
    print(faust_code)

    # --- Compile Faust to JAX ---
    DSP = fj.faust2jax(faust_code)(SAMPLE_RATE)
    instrument_jit = jax.jit(
        lambda params, noise, rate: DSP.apply(params, noise, rate, mutable="intermediates"),
        static_argnums=[2]
    )
    # --- Initialize DSP ---
    key = jrandom.PRNGKey(10)
    noise = jrandom.uniform(jrandom.PRNGKey(20), (DSP.getNumInputs(), SAMPLE_RATE), minval=-1, maxval=1)
    DSP_params = DSP.init(key, noise, SAMPLE_RATE)
    print(DSP_params)

    # --- Generate Target Sound and Play ---
    mo.output.clear()
    target_sound, _ = fj.process_noise_in_faust(faust_code, key, length_seconds=length_seconds)
    fj.show_audio(target_sound)

    # --- Parameter Sweep ---
    target_param = list(DSP_params["params"].keys())[0]
    param_linspace = jnp.linspace(-0.99, 1.0, 300, endpoint=False)
    programs = [
        {**DSP_params, "params": {**DSP_params["params"], target_param: x}}
        for x in param_linspace
    ]
    init_p = programs[50]
    # --- Render and Play a Swept Version ---
    s, _ = instrument_jit(init_p, noise, SAMPLE_RATE)
    fj.show_audio(s)


    return DSP_params, instrument_jit, noise, target_param, target_sound


@app.cell
def _(DSP_params):
    DSP_params
    return


@app.cell
def _(
    SAMPLE_RATE,
    clip_spec,
    dm_pix,
    dtw_jax,
    instrument_jit,
    kernel,
    loss_multi_spec,
    naive_loss,
    noise,
    onset_1d,
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
    return


@app.cell
def _():
    # # Optimizer setup
    # learning_rate = 0.04
    # num_steps = 100

    # optimizer = optax.rmsprop(learning_rate)
    # opt_state = optimizer.init(init_p)

    # loss_history = []
    # param_history = []
    # grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


    # @jax.jit
    # def update(params, opt_state):
    #     (loss, pred), grads = grad_fn(params)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     new_params = optax.apply_updates(params, updates)
    #     return new_params, opt_state, loss

    # params = init_p
    # for step in range(num_steps):
    #     params, opt_state, loss = update(params, opt_state)

    #     loss_history.append(float(loss))
    #     param_history.append(float(params["params"][target_param]))

    #     if step % 10 == 0:
    #         print(f"Step {step:3d} | Loss: {loss:.6f} | Param: {param_history[-1]:.4f}")

    # true_param_value = float(DSP_params["params"][target_param])

    # plt.figure(figsize=(12, 4))

    # # --- Loss plot ---
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_history)
    # plt.title("Loss over Time")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")

    # # --- Parameter evolution plot ---
    # plt.subplot(1, 2, 2)
    # plt.plot(param_history, label="Optimized Value")
    # plt.axhline(y=true_param_value, color='red', linestyle='--', linewidth=2, label="Target Value")
    # plt.title("Parameter Value over Time")
    # plt.xlabel("Step")
    # plt.ylabel(f"Param: {target_param}")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return


@app.cell
def _(DSP_params):
    DSP_params
    return


@app.cell
def _(
    DSP_params,
    SAMPLE_RATE,
    instrument_jit,
    jax,
    jnp,
    naive_loss,
    noise,
    plt,
    spec_func,
    target_param,
    target_sound,
):
    # import jax
    # import jax.numpy as jnp
    # import random
    # import optax
    # import matplotlib.pyplot as plt

    # Assume these are already defined:
    # - instrument_jit
    # - target_sound
    # - noise
    # - SAMPLE_RATE
    # - spec_func
    # - naive_loss
    # - DSP_params
    # - target_param

    # Initialize mean and std over the parameter
    mean = jnp.array([0.0])
    std = jnp.array([0.5])
    learning_rate = 0.3
    key1 = jax.random.PRNGKey(0)

    # Get true value to compare
    true_value = float(DSP_params.copy()["params"][target_param])

    print(DSP_params)
    # Reward = negative L1 spectral loss
    params = DSP_params.copy()

    def reward_fn(param_val): 
        params["params"][target_param] = param_val[0]
        pred = instrument_jit(params, noise, SAMPLE_RATE)[0]
        return -naive_loss(spec_func(pred)[0], spec_func(target_sound))

    # Sample from policy
    def sample_params(key, mean, std):
        return mean + std * jax.random.normal(key1, shape=mean.shape)

    # REINFORCE update
    @jax.jit
    def reinforce_update(mean, std, key1):
        print(DSP_params)
        sampled = sample_params(key1, mean, std)
        reward = reward_fn(sampled)

        grad_mean = (sampled - mean) / (std**2) * reward
        grad_std = ((sampled - mean)**2 - std**2) / (std**3) * reward

        new_mean = mean + learning_rate * grad_mean
        new_std = std + learning_rate * grad_std
        return new_mean, new_std, reward, sampled

    # Optimization loop
    rewards = []
    param_vals = []

    for step in range(1000):
        key1, subkey = jax.random.split(key1)
        mean, std, reward, sampled = reinforce_update(mean, std, subkey)
        rewards.append(float(reward))
        param_vals.append(float(sampled[0]))
        if step % 10 == 0:
            print(f"Step {step:3d} | Param: {sampled[0]:.4f} | Reward: {reward:.6f}")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Reward over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(param_vals, label="Sampled Param")
    plt.axhline(true_value, color='red', linestyle='--', label='True Param')
    plt.title("Parameter over Time")
    plt.xlabel("Step")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    app.run()

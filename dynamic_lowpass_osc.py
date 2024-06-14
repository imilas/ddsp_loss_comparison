import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import functools
    from functools import partial
    import itertools
    from pathlib import Path
    import os
    import jax
    import jax.numpy as jnp
    from jax import random

    from flax import linen as nn
    from flax.training import train_state  # Useful dataclass to keep train state
    from flax.core.frozen_dict import unfreeze
    import optax

    from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
    import dawdreamer.faust.box as fbox

    from tqdm.notebook import tqdm
    import time
    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc

    from IPython.display import HTML
    from IPython.display import Audio
    import IPython.display as ipd

    from kymatio.jax import Scattering1D

    default_device = "cpu"  # or 'gpu'
    jax.config.update("jax_platform_name", default_device)
    return (
        Audio,
        FaustContext,
        HTML,
        Path,
        Scattering1D,
        animation,
        createLibContext,
        default_device,
        destroyLibContext,
        fbox,
        functools,
        ipd,
        itertools,
        jax,
        jnp,
        librosa,
        mo,
        nn,
        np,
        optax,
        os,
        partial,
        plt,
        random,
        rc,
        time,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(Audio, ipd):
    SAMPLE_RATE = 44100


    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.0:
            data /= abs(data).max()
        ipd.display(
            Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)
        )
    return SAMPLE_RATE, show_audio


@app.cell
def __(FaustContext, SAMPLE_RATE, fbox, jnp):
    def make_sine(freq: float, T: int, sr=SAMPLE_RATE):
        """Return sine wave based on freq in Hz and duration T in samples"""
        return jnp.sin(jnp.pi * 2.0 * freq * jnp.arange(T) / sr)


    faust_code = f"""
    import("stdfaust.lib");
    cutoff = hslider("cutoff", 440., 20., 20000., .01);
    FX = fi.lowpass(1, cutoff);
    replace = !,_;
    process = ["cutoff":replace -> FX];
    """

    module_name = "MyDSP"

    with FaustContext():

        box = fbox.boxFromDSP(faust_code)

        jax_code = fbox.boxToSource(
            box, "jax", module_name, ["-a", "jax/minimal.py"]
        )

    custom_globals = {}
    exec(jax_code, custom_globals)
    FilterModel = custom_globals[module_name]
    return (
        FilterModel,
        box,
        custom_globals,
        faust_code,
        jax_code,
        make_sine,
        module_name,
    )


@app.cell
def __(FilterModel, SAMPLE_RATE, jax, jnp, make_sine, nn, np, partial):
    # We don't vmap FilterModel in-place because we need the original version inside AutomationModel
    HiddenModel = nn.vmap(
        FilterModel,
        in_axes=(0, None),
        variable_axes={"params": None},
        split_rngs={"params": False},
    )

    hidden_model = HiddenModel(SAMPLE_RATE)


    class AutomationModel_(nn.Module):

        automation_samples: int

        def getNumInputs(self):
            return 1

        @nn.compact
        def __call__(self, x, T: int) -> jnp.array:
            # make the learnable cutoff automation parameter.
            freq = self.param("freq", nn.initializers.constant(3), 1)
            angles = jnp.linspace(0, 1, T, endpoint=False)
            automation = jnp.sin(freq * angles * 2 * np.pi)
            automation = jnp.expand_dims(automation, axis=0)
            automation = jnp.interp(
                automation, jnp.array([-1, 1]), jnp.array([20, 20000])
            )
            self.sow("intermediates", "freq", freq)
            self.sow("intermediates", "cutoff", automation)

            x = jnp.concatenate([automation, x], axis=-2)
            filterModel = FilterModel(sample_rate=SAMPLE_RATE)
            audio = filterModel(x, T)

            return audio


    # set control rate to 100th of audio rate
    AUTOMATION_DOWNSAMPLE = 100  # @param {type:"integer"}
    RECORD_DURATION = 1.0  # @param {type:"number"}
    T = int(RECORD_DURATION * SAMPLE_RATE)
    automation_samples = T // AUTOMATION_DOWNSAMPLE

    jit_inference_fn = jax.jit(
        partial(
            AutomationModel_(automation_samples=automation_samples).apply,
            mutable="intermediates",
        ),
        static_argnums=[2],
    )
    AutomationModel = nn.vmap(
        AutomationModel_,
        in_axes=(0, None),
        variable_axes={"params": None},
        split_rngs={"params": False},
    )

    train_model = AutomationModel(automation_samples=automation_samples)

    batch_size = 1  # @param {type: 'integer'}
    hidden_automation_freq = 10.0  # @param {type:"number"}
    hidden_automation = 10_000 + make_sine(hidden_automation_freq, T) * 9500
    jnp.expand_dims(hidden_automation, axis=0)
    hidden_automation = jnp.tile(hidden_automation, (batch_size, 1, 1))
    print("hidden_automation shape: ", hidden_automation.shape)
    return (
        AUTOMATION_DOWNSAMPLE,
        AutomationModel,
        AutomationModel_,
        HiddenModel,
        RECORD_DURATION,
        T,
        automation_samples,
        batch_size,
        hidden_automation,
        hidden_automation_freq,
        hidden_model,
        jit_inference_fn,
        train_model,
    )


@app.cell
def __(mo):
    mo.md(
        """
    ```FilterModel```: Takes in a series of values which control the cutoff parameter. These values (or automation) is the first parameter, and audio is the second. 

    ```HiddenModel```: Is a vmapped FilterModel

    ```AutomationModel```: Has parameter 'freq' which determines the cutoff oscillation frequency. It takes in x/audio and T as arguments. Is vmapped in placed.

    ```jit_inference_fn```: Partially jitted AutomationModel for inference. Is not vmapped? 

    ``` train_model```: instance of Automation model, with static argument set. 

    ``` hidden_automation_*```: true values we are approximating 

    """
    )
    return


@app.cell
def __(
    T,
    batch_size,
    hidden_automation,
    hidden_model,
    jnp,
    random,
    train_model,
):
    key = random.PRNGKey(420)
    input_shape = (batch_size, train_model.getNumInputs(), T)

    key, subkey = random.split(key)
    hidden_inputs = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

    hidden_inputs = jnp.concatenate([hidden_automation, hidden_inputs], axis=1)
    print("hidden_shape:", hidden_inputs.shape)
    key, subkey = random.split(key)
    hidden_params = hidden_model.init({"params": subkey}, hidden_inputs, T)
    print("hidden params", hidden_params)
    key, subkey = random.split(key)
    rand_x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

    train_inputs = rand_x
    print("train_inputs shape:", train_inputs.shape)

    key, subkey = random.split(key)
    train_params = train_model.init({"params": subkey}, train_inputs, T)
    # print("train params:", train_params["params"].keys())
    print(" train params shape:", train_params["params"])
    return (
        hidden_inputs,
        hidden_params,
        input_shape,
        key,
        rand_x,
        subkey,
        train_inputs,
        train_params,
    )


@app.cell
def __(
    T,
    hidden_inputs,
    hidden_model,
    show_audio,
    train_inputs,
    train_model,
    train_params,
):
    hidden_sounds = hidden_model.apply({}, hidden_inputs, T)
    train_sounds = train_model.apply(train_params, train_inputs, T)
    show_audio(hidden_sounds[0])
    show_audio(train_sounds[0])
    return hidden_sounds, train_sounds


@app.cell
def __(T, jnp, partial, train_sounds):
    from audax.core import functional

    NFFT = 256
    WIN_LEN = 400
    HOP_LEN = 160

    # creates a spectrogram helper
    window = jnp.hanning(WIN_LEN)
    spec_func = partial(
        functional.spectrogram,
        pad=0,
        window=window,
        n_fft=NFFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        power=2.0,
        normalized=False,
        center=True,
        onesided=True,
    )
    fb = functional.melscale_fbanks(
        n_freqs=(NFFT // 2) + 1, n_mels=64, sample_rate=T, f_min=60.0, f_max=7800.0
    )
    mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)
    jax_spec = spec_func(train_sounds[0][0])
    mel_spec = mel_spec_func(jax_spec)  # output of shape (1, 101, 64)
    return (
        HOP_LEN,
        NFFT,
        WIN_LEN,
        fb,
        functional,
        jax_spec,
        mel_spec,
        mel_spec_func,
        spec_func,
        window,
    )


@app.cell
def __(jax_spec, plt):
    plt.imshow(
        jax_spec[0].T,
        aspect=2,
        cmap=plt.cm.gray_r,
        origin="lower",
    )
    # plt.imshow(
    #     spec_func(train_sounds[0])[0].T,
    #     aspect=2,
    #     cmap=plt.cm.gray_r,
    #     origin="lower",
    # )
    plt.ylabel("FFT Bin Number")
    plt.xlabel("Frame Number")
    plt.title("Spectrograms")
    return


@app.cell
def __(mo):
    mo.md(
        """here we use the fourier representation of audio to calculate the loss. The loss function is $\sum |d_{i,j}|$ where $d = \chi_x - \chi_y$, $\chi$ is the spectrogram representation, and $x$ is the output of the synthesizer, and $y$ is the sound we're approximating. The dimensions $i$ and $j$ depend on the length of the sound and the parameters of the FFT function.

    The issue with this representation is that d is sensetive to the alignment of sounds. If x is a copy of y but offset by a few time-steps, the loss function would return a large value. 

    An improvement to this setup could be to convolve the loss functions and return the largest value as a similarity metric. 
    """
    )
    return


@app.cell
def __(
    T,
    hidden_automation,
    hidden_model,
    hidden_params,
    input_shape,
    jax,
    jit_inference_fn,
    jnp,
    key,
    np,
    optax,
    random,
    spec_func,
    tqdm,
    train_model,
    train_params,
    train_state,
):
    learning_rate = 2  # @param {type: 'number'}
    momentum = 0.95  # @param {type: 'number'}

    # Create Train state
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=train_model.apply, params=train_params, tx=tx
    )


    @jax.jit
    def train_step(state, x, y):
        """Train for a single step."""

        def loss_fn(params):
            pred = train_model.apply(params, x, T)

            ## spec loss
            pred_t = spec_func(pred[0][0])
            y_t = spec_func(y[0][0])
            loss = (jnp.abs(pred_t - y_t)).mean()

            # # L1 time-domain loss
            # loss = (jnp.abs(pred - y)).mean()

            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss


    losses = []
    cutoffs = []

    train_steps = 200
    steps_per_eval = 1

    pbar = tqdm(range(train_steps))

    jit_hidden_model = jax.jit(hidden_model.apply, static_argnums=[2])

    # Data containers for animation
    cutoff_data = []
    freq_data = []
    for n in pbar:
        print(n, end="\r")
        key1, subkey1 = random.split(key)
        # generate random signal to be filtered
        x = random.uniform(subkey1, shape=input_shape, minval=-1, maxval=1)

        # concat hidden automation to random signal
        automation_and_x = jnp.concatenate([hidden_automation, x], axis=1)

        # get ground truth of filtered signal
        y = jit_hidden_model(hidden_params, automation_and_x, T)

        # optimize learned automation based on input signal and GT filtered signal
        state, loss = train_step(state, x, y)

        if n % steps_per_eval == 0:
            audio, mod_vars = jit_inference_fn(state.params, x[0], T)
            cutoff = np.array(mod_vars["intermediates"]["cutoff"])[0]
            cutoff_data.append(cutoff)
            freq_data.append(mod_vars["intermediates"]["freq"][0])
            pbar.set_description(f"loss: {loss:.2f}")
            losses.append(loss)
    return (
        audio,
        automation_and_x,
        cutoff,
        cutoff_data,
        cutoffs,
        freq_data,
        jit_hidden_model,
        key1,
        learning_rate,
        loss,
        losses,
        mod_vars,
        momentum,
        n,
        pbar,
        state,
        steps_per_eval,
        subkey1,
        train_step,
        train_steps,
        tx,
        x,
        y,
    )


@app.cell
def __(freq_data, losses, plt):
    fig_1, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("loss", color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "OSC-frequency", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(freq_data, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    # ax2.set_yscale("log")
    fig_1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return ax1, ax2, color, fig_1


@app.cell
def __(T, jnp, spec_func, train_model, x, y):
    # loss landscape
    def loss_fn_sigdiff(params):
        pred = train_model.apply(params, x, T)

        ## spec loss
        pred_t = spec_func(pred[0][0])
        y_t = spec_func(y[0][0])
        loss = (jnp.abs(pred_t - y_t)).mean()

        # # L1 time-domain loss
        # loss = (jnp.abs(pred - y)).mean()
        return loss
    return loss_fn_sigdiff,


@app.cell
def __(freq_data, jax, jnp, loss_fn_sigdiff, np):
    def test_freq(f):
        return loss_fn_sigdiff({"params": {"freq": jnp.array([f])}})


    test_freqs = jax.vmap(test_freq, 0)
    test_freqs_jitted = jax.jit(test_freqs)
    freqs_explored = [np.min(freq_data) - 40, np.max([np.max(freq_data) + 1, 13])]
    oscillation_frequencies = jnp.linspace(
        freqs_explored[0], freqs_explored[1], 1000
    )
    test_losses = test_freqs_jitted(oscillation_frequencies)
    return (
        freqs_explored,
        oscillation_frequencies,
        test_freq,
        test_freqs,
        test_freqs_jitted,
        test_losses,
    )


@app.cell
def __(
    HTML,
    animation,
    freq_data,
    freqs_explored,
    losses,
    mo,
    np,
    oscillation_frequencies,
    plt,
    rc,
    test_losses,
):
    mo.output.clear()
    # Initialize the plot for animation
    fig, ax = plt.subplots(figsize=(8, 4))

    (line1,) = ax.plot([], [], label="loss landscape")
    scat1 = ax.scatter([], [], label="prediction")
    scat2 = ax.scatter([], [], label="optimal point")
    ax.set_title("Optimizing a Lowpass Filter's Cutoff")
    ax.set_ylabel("loss")
    ax.set_xlabel("oscillator frequency")
    ax.set_ylim(0, np.max(test_losses))
    ax.set_xlim(freqs_explored)
    plt.legend(loc="best")
    time_axis = oscillation_frequencies


    # Function to update the plot for each frame
    def update_plot(frame):
        line1.set_data(time_axis, test_losses)
        scat1.set_offsets([freq_data[frame][0], losses[frame]])
        scat2.set_offsets([10, np.min(test_losses)])
        return line1, scat1


    # Creating the animation
    rc("animation", html="jshtml")
    anim = animation.FuncAnimation(fig, update_plot, frames=len(losses), blit=True)
    HTML(anim.to_html5_video())
    return anim, ax, fig, line1, scat1, scat2, time_axis, update_plot


@app.cell
def __(mo):
    mo.md(
        """
        The parameter search is difficult. How we define loss leads to a landscape that oscillates quite a bit, making it easy to get stuck in local minimums. 

        If we were modifying the synthesizer parameters ourselves, we would search for the desired parameters by incrementally changing the synthesizer's parameters, listening to the changes in the output sound, and deciding how to change the parameter based on what we heard. 

        Here we try a map-elites algorithm


        """
    )
    return


@app.cell
def __():
    from brax.v1.envs.env import State
    import qdax
    from qdax.core.containers.mapelites_repertoire import (
        compute_cvt_centroids,
        MapElitesRepertoire,
    )
    from qdax.core.emitters.mutation_operators import isoline_variation
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.core.neuroevolution.buffers.buffer import QDTransition
    from qdax.environments import QDEnv
    from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
    from qdax.types import Descriptor
    from qdax.utils.metrics import CSVLogger, default_qd_metrics
    from qdax.utils.plotting import plot_map_elites_results
    from qdax.utils.plotting import plot_multidimensional_map_elites_grid
    return (
        CSVLogger,
        Descriptor,
        MAPElites,
        MapElitesRepertoire,
        MixingEmitter,
        QDEnv,
        QDTransition,
        State,
        compute_cvt_centroids,
        default_qd_metrics,
        isoline_variation,
        plot_map_elites_results,
        plot_multidimensional_map_elites_grid,
        qdax,
        scoring_function,
    )


@app.cell
def __():
    # @title QD Training Definitions Fields
    # batch_size = 6  #@param {type:"integer"}
    episode_length = 1
    num_iterations = 8000  # @param {type:"integer"}
    log_period = 20
    seed = 42  # @param {type:"integer"}
    iso_sigma = 0.005  # @param {type:"number"}
    line_sigma = 0.01  # @param {type:"number"}
    num_init_cvt_samples = 50000  # @param {type:"integer"}
    num_centroids = 32  # @param {type:"integer"}
    reward_offset = 1e-6  # minimum reward value to make sure qd_score are positive
    return (
        episode_length,
        iso_sigma,
        line_sigma,
        log_period,
        num_centroids,
        num_init_cvt_samples,
        num_iterations,
        reward_offset,
        seed,
    )


@app.cell
def __(
    AutomationModel,
    AutomationModel_,
    SAMPLE_RATE,
    automation_samples,
    jax,
    jnp,
    random,
    train_model,
):
    N_CHANNELS = train_model.getNumInputs()
    print("N_CHANNELS:", N_CHANNELS)
    N_SECONDS = 1.0  # @param {type:"number"}
    N_SAMPLES = int(SAMPLE_RATE * N_SECONDS)

    key2 = random.PRNGKey(0)
    model = AutomationModel_(automation_samples=automation_samples)
    target_variables = model.init(
        {"params": key2}, jnp.ones((N_CHANNELS, N_SAMPLES)), N_SAMPLES
    )
    print("target_variables:", target_variables)

    # Our behavior descriptor length is the number of parameters.
    num_parameters = len(target_variables["params"].keys())
    behavior_descriptor_length = num_parameters
    print("behavior_descriptor_length:", behavior_descriptor_length)

    # show the target intermediate parameters which we want our trained model to discover.
    x_qdax = -1.0 + 2.0 * jax.random.uniform(key2, (N_CHANNELS, N_SAMPLES))
    y_qdax, target_parameters = model.apply(
        target_variables, x_qdax, N_SAMPLES, mutable="intermediates"
    )
    print("target_parameters:", target_parameters["intermediates"])

    # batch_model = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': True})

    batch_model = AutomationModel


    env_inference_fn = jax.jit(model.apply, static_argnums=[2])
    # env_inference_fn(target_variables,x_qdax,T)
    return (
        N_CHANNELS,
        N_SAMPLES,
        N_SECONDS,
        batch_model,
        behavior_descriptor_length,
        env_inference_fn,
        key2,
        model,
        num_parameters,
        target_parameters,
        target_variables,
        x_qdax,
        y_qdax,
    )


@app.cell
def __(mo):
    mo.md(
        """
    env_inference_fn takes a set of parameters, an array of audio, and a static argument determining the sample rate. 

    model is an instance of the AutomationModel_ 

    """
    )
    return


@app.cell
def __(
    Dict,
    List,
    N_CHANNELS,
    N_SAMPLES,
    QDEnv,
    State,
    Tuple,
    env_inference_fn,
    jax,
    jnp,
    reward_offset,
    target_variables,
):
    # @title Define a Custom Environment
    class MyEnv(QDEnv):

        @property
        def state_descriptor_length(self) -> int:
            # note: this value doesn't matter for us
            pass

        @property
        def state_descriptor_name(self) -> str:
            # note: this value doesn't matter for us
            pass

        @property
        def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
            # note: this value doesn't matter for us
            pass

        @property
        def behavior_descriptor_length(self) -> int:
            return behavior_descriptor_length

        @property
        def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
            # When we converted Faust to JAX, all of the parameters were normalized to [-1., 1]
            a_min = [-1.0 for _ in range(self.behavior_descriptor_length)]
            a_max = [1.0 for _ in range(self.behavior_descriptor_length)]
            return a_min, a_max

        @property
        def name(self) -> str:
            return "MyEnvFoo"

        @property
        def observation_size(self):
            # note: this value doesn't matter for us
            pass

        @property
        def action_size(self) -> int:
            # note: this value doesn't matter for us
            pass

        def reset(self, rng: jnp.ndarray) -> State:
            """Resets the environment to an initial state."""
            x = -1 + 2.0 * jax.random.uniform(rng, (N_CHANNELS, N_SAMPLES))

            # use our ground-truth variables to generate audio
            y = env_inference_fn(target_variables, x, N_SAMPLES)

            obs_init = {"x": x, "y": y}
            reward, done = jnp.zeros(2)
            metrics: Dict = {}
            info_init = {"state_descriptor": obs_init}
            return State(None, obs_init, reward, done, metrics, info_init)

        def step(self, state: State, actions) -> State:
            """Run one timestep of the environment's dynamics."""

            x, y = state.obs["x"], state.obs["y"]

            # # use actions (learned parameters) to generate audio prediction
            # print(actions["params"]["freq"].shape)
            # print(target_variables["params"]["freq"].shape)

            action_value = actions["params"]["freq"]
            cutoff = {"params": {"freq": jnp.array([action_value])}}

            pred = env_inference_fn(cutoff, x, N_SAMPLES)

            # L1 time-domain loss
            loss = jnp.abs(pred - y).mean()

            # keep reward positive so that the QD-score works correctly.
            reward = jnp.maximum(reward_offset, 1.0 - loss)

            done = jnp.array(1.0)

            new_obs = state.obs
            # update the state
            return state.replace(obs=new_obs, reward=reward, done=done)  # type: ignore
    return MyEnv,


@app.cell
def __(
    MyEnv,
    N_CHANNELS,
    N_SAMPLES,
    automation_samples,
    batch_model,
    batch_size,
    jax,
    jnp,
    key2,
    random,
):
    # @title Initialize environment, and population params
    # Init environment
    my_env = MyEnv(config=None)

    # Create the initial environment states
    random_key, key3 = jax.random.split(key2)
    keys2 = jnp.repeat(jnp.expand_dims(key3, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(my_env.reset))
    init_states = reset_fn(keys2)

    # Randomly initialize the parameters:
    # Pass a fake batch to get init_variables.
    fake_batch = jnp.zeros((batch_size, N_CHANNELS, N_SAMPLES), jnp.float32)
    random_key, key4 = jax.random.split(random_key)

    init_variables = batch_model(automation_samples=automation_samples).init(
        key4, fake_batch, N_SAMPLES
    )
    # The init_variables start as the ground truth,
    # so we need to randomize them between -1 and 1
    random_key, key5 = jax.random.split(random_key)
    init_variables = jax.tree_util.tree_map(
        lambda x: random.uniform(
            key5, x.shape, minval=-1, maxval=1, dtype=jnp.float32
        ),
        init_variables,
    )
    return (
        fake_batch,
        init_states,
        init_variables,
        key3,
        key4,
        key5,
        keys2,
        my_env,
        random_key,
        reset_fn,
    )


@app.cell
def __(QDTransition, jax, my_env):
    # @title Define the function to play a step with the policy in the environment
    @jax.jit
    def play_step_fn(
        env_state,
        policy_params,
        random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """
        state_desc = env_state.info["state_descriptor"]
        next_state = my_env.step(env_state, policy_params)

        truncations = None
        next_state_desc = next_state.info["state_descriptor"]

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=policy_params,
            truncations=truncations,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

        return next_state, policy_params, random_key, transition
    return play_step_fn,


@app.cell
def __(
    Descriptor,
    QDTransition,
    behavior_descriptor_length,
    default_qd_metrics,
    episode_length,
    functools,
    init_states,
    jnp,
    play_step_fn,
    reward_offset,
    scoring_function,
):
    # @title Define the scoring function and the way metrics are computed


    # Prepare the scoring function
    def bd_extraction_fn(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
        matrix = jnp.concatenate(list(data.actions["params"].values()), axis=-1)

        # No matter what, this assertion should be True
        assert matrix.shape[-1] == behavior_descriptor_length

        return matrix


    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )
    return bd_extraction_fn, metrics_function, scoring_fn


@app.cell
def __(
    MixingEmitter,
    batch_size,
    functools,
    iso_sigma,
    isoline_variation,
    line_sigma,
):
    # @title Define the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    return mixing_emitter, variation_fn


@app.cell
def __(
    MAPElites,
    compute_cvt_centroids,
    key5,
    metrics_function,
    mixing_emitter,
    my_env,
    num_centroids,
    num_init_cvt_samples,
    scoring_fn,
):
    # @title Instantiate and initialize the MAP-Elites algorithm

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, key6 = compute_cvt_centroids(
        num_descriptors=my_env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=-1.0,
        maxval=1.0,
        random_key=key5,
    )
    return centroids, key6, map_elites


@app.cell
def __(
    CSVLogger,
    centroids,
    functools,
    init_variables,
    jax,
    jnp,
    key6,
    log_period,
    map_elites,
    num_iterations,
    os,
    pbar,
    time,
):
    # Compute initial repertoire and emitter state
    repertoire, emitter_state, key7 = map_elites.init(
        init_variables, centroids, key6
    )
    # @title Launch MAP-Elites iterations
    num_loops = int(num_iterations / log_period)

    csv_logger = CSVLogger(
        "mapelites-logs.csv",
        header=[
            "loop",
            "iteration",
            "qd_score",
            "max_fitness",
            "coverage",
            "time",
        ],
    )
    all_metrics = {}

    jit_scan = jax.jit(
        functools.partial(
            jax.lax.scan, map_elites.scan_update, xs=(), length=log_period
        )
    )

    # pbar = tqdm(range(num_loops))
    for i in pbar:
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key8,
        ), metrics = jit_scan(
            init=(repertoire, emitter_state, key7),
        )
        timelapse = time.time() - start_time

        # log metrics
        logged_metrics = {
            "time": timelapse,
            "loop": 1 + i,
            "iteration": 1 + i * log_period,
        }
        for k, value in metrics.items():
            # take last value
            logged_metrics[k] = value[-1]

            # take all values
            if k in all_metrics.keys():
                all_metrics[k] = jnp.concatenate([all_metrics[k], value])
            else:
                all_metrics[k] = value

        max_fitness = "{:.4f}".format(metrics["max_fitness"].max())
        pbar.set_description(f"Max Fitness: {max_fitness}")
        csv_logger.log(logged_metrics)

    # Save the repertoire
    repertoire_path = "./last_repertoire/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)
    return (
        all_metrics,
        csv_logger,
        emitter_state,
        i,
        jit_scan,
        k,
        key7,
        key8,
        logged_metrics,
        max_fitness,
        metrics,
        num_loops,
        repertoire,
        repertoire_path,
        start_time,
        timelapse,
        value,
    )


@app.cell
def __(
    MapElitesRepertoire,
    N_CHANNELS,
    N_SAMPLES,
    jax,
    jnp,
    key8,
    model,
    np,
    repertoire_path,
):
    # @title Loading the Repertoire

    # Init population of policies
    subkey2, key9 = jax.random.split(key8)
    fake_params = model.init(
        key9, jnp.float32(np.random.random((N_CHANNELS, N_SAMPLES))), N_SAMPLES
    )

    _, reconstruction_fn = jax.flatten_util.ravel_pytree(fake_params)

    recons_repertoire = MapElitesRepertoire.load(
        reconstruction_fn=reconstruction_fn, path=repertoire_path
    )
    return fake_params, key9, recons_repertoire, reconstruction_fn, subkey2


@app.cell
def __(jnp, repertoire):
    # @title Show the Best Parameters
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_fitness = jnp.max(repertoire.fitnesses)
    best_bd = repertoire.descriptors[best_idx]

    print(
        f"Best fitness in the repertoire: {best_fitness:.4f}\n",
        f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
        f"Index in the repertoire of this individual: {best_idx}\n",
    )
    return best_bd, best_fitness, best_idx


@app.cell
def __(
    N_SAMPLES,
    best_idx,
    jax,
    jit_inference_fn,
    jnp,
    my_env,
    repertoire,
    x,
):
    my_params = jax.tree_util.tree_map(lambda x: x[best_idx], repertoire.genotypes)

    jit_env_reset = jax.jit(my_env.reset)
    jit_env_step = jax.jit(my_env.step)

    rng = jax.random.PRNGKey(seed=1)
    state_2 = jit_env_reset(rng=rng)

    action, mod_vars_2 = jit_inference_fn(
        {"params": {"freq": jnp.array([0.67911553])}}, x[0], N_SAMPLES
    )


    # Note that our optimal parameters could have put the parametric EQ sections
    # in the wrong order.
    # Linear time invariant filters have this commutativity property.
    # We can do some permutation tricks to make the predicted parameters line up
    # with the ground truths.

    # If you can think through the permutations yourself,
    # just print out the variables and visually inspect them:
    print(mod_vars_2["intermediates"])
    return (
        action,
        jit_env_reset,
        jit_env_step,
        mod_vars_2,
        my_params,
        rng,
        state_2,
    )


@app.cell
def __(action, show_audio):
    show_audio(action)
    return


if __name__ == "__main__":
    app.run()

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
    from helpers import qdax_plot
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
        qdax_plot,
        random,
        rc,
        time,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


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
def __(Audio, ipd, np, wavfile):
    SAMPLE_RATE = 44100


    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.0:
            data /= abs(data).max()
        ipd.display(
            Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)
        )


    samplerate, goal_sound = wavfile.read(
        "./sounds/target_sounds/two_osc_400_7.wav"
    )
    target = goal_sound[0:SAMPLE_RATE, 0]
    target = np.array(target, dtype="float")
    target = target / abs(target).max()
    return SAMPLE_RATE, goal_sound, samplerate, show_audio, target


@app.cell
def __():
    # @title QD Training Definitions Fields
    batch_size = 3  # @param {type:"integer"}
    episode_length = 1
    num_iterations = 1000  # @param {type:"integer"}
    log_period = 10
    seed = 420  # @param {type:"integer"}
    iso_sigma = 0.1  # @param {type:"number"}
    line_sigma = 0.1  # @param {type:"number"}
    num_init_cvt_samples = 50000  # @param {type:"integer"}
    num_centroids = 16  # @param {type:"integer"}
    reward_offset = 1e-6  # minimum reward value to make sure qd_score are positive
    return (
        batch_size,
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
def __(FaustContext, SAMPLE_RATE, fbox, jax):
    # @title Define Target JAX Model and Ground Truth Parameters
    faust_code = f"""
    import("stdfaust.lib");
    cutoff = hslider("cutoff",500,101,1000,1);
    osc_f = hslider("osc_f",10,1,100,0.5);
    //osc_mag = hslider("osc_mag",200,10,400,1);
    FX = fi.lowpass(5,cutoff);
    process = os.osc(osc_f)*400,_:["cutoff":+(_,_)->FX];
    """
    module_name = "MyDSP"
    with FaustContext():
        model_source_code = fbox.boxToSource(
            fbox.boxFromDSP(faust_code),
            "jax",
            module_name,
            ["-a", "jax/minimal.py"],
        )
    custom_globals = {}
    exec(model_source_code, custom_globals)  # security risk!
    MyDSP = custom_globals[module_name]
    model = MyDSP(SAMPLE_RATE)
    env_inference_fn = jax.jit(model.apply, static_argnums=[2])
    # The number of input channels the model takes.
    N_CHANNELS = model.getNumInputs()
    print("N_CHANNELS:", N_CHANNELS)
    # what is the difference between model, batch_model, MyDSP, env_inference_fn
    return (
        MyDSP,
        N_CHANNELS,
        custom_globals,
        env_inference_fn,
        faust_code,
        model,
        model_source_code,
        module_name,
    )


@app.cell
def __(
    MyDSP,
    N_CHANNELS,
    SAMPLE_RATE,
    batch_size,
    env_inference_fn,
    jax,
    jnp,
    model,
    nn,
    random,
):
    # @title Initialize Batched Model for Training

    N_SECONDS = 1.0  # @param {type:"number"}
    N_SAMPLES = int(SAMPLE_RATE * N_SECONDS)

    init_key = random.PRNGKey(0)
    target_variables = model.init(
        {"params": init_key},
        jnp.ones((N_CHANNELS, N_SAMPLES), jnp.float32),
        N_SAMPLES,
    )
    print("target_variables:", target_variables)
    # Our behavior descriptor length is the number of parameters.
    num_parameters = len(target_variables["params"].keys())
    behavior_descriptor_length = num_parameters
    # behavior_descriptor_length = 2 # todo
    print("behavior_descriptor_length:", behavior_descriptor_length)

    # show the target intermediate parameters which we want our trained model to discover.
    x = -1.0 + 2.0 * jax.random.uniform(init_key, (N_CHANNELS, N_SAMPLES))
    y, target_parameters = model.apply(
        target_variables, x, N_SAMPLES, mutable="intermediates"
    )
    print("target_parameters:", target_parameters["intermediates"])

    rand_key, subkey = jax.random.split(init_key)
    keys = jax.random.split(subkey, num=batch_size)
    batch_model = nn.vmap(
        MyDSP,
        in_axes=(0, None),
        variable_axes={"params": 0},
        split_rngs={"params": True},
    )

    # shape of outputs
    env_inference_fn(target_variables, x, N_SAMPLES).shape
    return (
        N_SAMPLES,
        N_SECONDS,
        batch_model,
        behavior_descriptor_length,
        init_key,
        keys,
        num_parameters,
        rand_key,
        subkey,
        target_parameters,
        target_variables,
        x,
        y,
    )


@app.cell
def __(
    N_SAMPLES,
    env_inference_fn,
    show_audio,
    target,
    target_variables,
    x,
):
    outputs = env_inference_fn(target_variables, x, N_SAMPLES)
    show_audio(x)
    show_audio(outputs[0])
    show_audio(target)
    return outputs,


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
    target,
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

            # use actions (learned parameters) to generate audio prediction
            pred = env_inference_fn(actions, x, N_SAMPLES)

            # L1 time-domain loss
            loss = jnp.abs(pred - target).mean()

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
    SAMPLE_RATE,
    batch_model,
    batch_size,
    jax,
    jnp,
    rand_key,
    random,
    subkey,
):
    # @title Initialize environment, and population params
    # Init environment
    my_env = MyEnv(config=None)
    # Create the initial environment states
    random_key2, subkey2 = jax.random.split(rand_key)
    keys2 = jnp.repeat(
        jnp.expand_dims(subkey2, axis=0), repeats=batch_size, axis=0
    )
    reset_fn = jax.jit(jax.vmap(my_env.reset))
    init_states = reset_fn(keys2)

    # Randomly initialize the parameters:
    # Pass a fake batch to get init_variables.
    fake_batch = jnp.zeros((batch_size, N_CHANNELS, N_SAMPLES), jnp.float32)
    init_variables = batch_model(SAMPLE_RATE).init(subkey2, fake_batch, N_SAMPLES)


    # The init_variables start as the ground truth,
    # so we need to randomize them between -1 and 1
    def random_split_like_tree(rng_key, target=None, treedef=None):
        if treedef is None:
            treedef = jax.tree_structure(target)
        keys = jax.random.split(rng_key, treedef.num_leaves)
        return jax.tree_unflatten(treedef, keys)


    keys_tree = random_split_like_tree(subkey, init_variables)

    init_variables = jax.tree_util.tree_map(
        lambda x, key_list: random.uniform(
            key_list, x.shape, minval=-1, maxval=1, dtype=jnp.float32
        ),
        init_variables,
        keys_tree,
    )
    return (
        fake_batch,
        init_states,
        init_variables,
        keys2,
        keys_tree,
        my_env,
        random_key2,
        random_split_like_tree,
        reset_fn,
        subkey2,
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
    metrics_function,
    mixing_emitter,
    my_env,
    num_centroids,
    num_init_cvt_samples,
    random_key2,
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
    centroids, random_key3 = compute_cvt_centroids(
        num_descriptors=my_env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=-1.0,
        maxval=1.0,
        random_key=random_key2,
    )
    return centroids, map_elites, random_key3


@app.cell
def __(
    CSVLogger,
    centroids,
    functools,
    init_variables,
    jax,
    jnp,
    log_period,
    map_elites,
    num_iterations,
    os,
    random_key3,
    time,
    tqdm,
):
    random_key4, subkey3 = jax.random.split(random_key3)
    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, subkey3
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

    pbar = tqdm(range(num_loops))
    for i in pbar:
        print("loop %d/%d" % (i, num_loops), end="\r")
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            random_key4,
        ), metrics = jit_scan(
            init=(repertoire, emitter_state, random_key4),
        )
        timelapse = time.time() - start_time
        # log metrics
        logged_metrics = {
            "time": timelapse,
            "loop": 1 + i,
            "iteration": 1 + i * log_period,
        }
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value
        # pbar.set_description(f"Max Fitness: {max_fitness}")
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
        key,
        logged_metrics,
        metrics,
        num_loops,
        pbar,
        random_key,
        random_key4,
        repertoire,
        repertoire_path,
        start_time,
        subkey3,
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
    model,
    np,
    random_key4,
    repertoire_path,
):
    # @title Loading the Repertoire

    # Init population of policies
    random_key5, subkey4 = jax.random.split(random_key4)
    fake_params = model.init(
        subkey4, jnp.float32(np.random.random((N_CHANNELS, N_SAMPLES))), N_SAMPLES
    )

    _, reconstruction_fn = jax.flatten_util.ravel_pytree(fake_params)

    repertoire_loaded = MapElitesRepertoire.load(
        reconstruction_fn=reconstruction_fn, path=repertoire_path
    )
    return (
        fake_params,
        random_key5,
        reconstruction_fn,
        repertoire_loaded,
        subkey4,
    )


@app.cell
def __(jnp, repertoire_loaded):
    # @title Show the Best Parameters
    best_idx = jnp.argmax(repertoire_loaded.fitnesses)
    best_fitness = jnp.max(repertoire_loaded.fitnesses)
    best_bd = repertoire_loaded.descriptors[best_idx]

    print(
        f"Best fitness in the repertoire: {best_fitness:.4f}\n",
        f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
        f"Index in the repertoire of this individual: {best_idx}\n",
    )
    return best_bd, best_fitness, best_idx


@app.cell
def __(SAMPLE_RATE, best_idx, jax, model, repertoire, show_audio, x):
    my_params = jax.tree_util.tree_map(lambda x: x[best_idx], repertoire.genotypes)

    best_outputs, best_found = model.apply(
        my_params, x, SAMPLE_RATE, mutable="intermediates"
    )
    show_audio(best_outputs[0])
    show_audio(best_outputs[1])
    return best_found, best_outputs, my_params


@app.cell
def __(best_found):
    best_found
    return


@app.cell
def __(
    all_metrics,
    batch_size,
    episode_length,
    jnp,
    num_iterations,
    plot_map_elites_results,
    repertoire,
):
    # @title Plotting

    # Create the x-axis array
    env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    # Note that our MAP-Elites Grid will look sparse when the `num_centroids` hyperparameter is small.
    fig2, axes2 = plot_map_elites_results(
        env_steps=env_steps,
        metrics=all_metrics,
        repertoire=repertoire,
        min_bd=-1,
        max_bd=1,
    )
    fig2
    return axes2, env_steps, fig2


if __name__ == "__main__":
    app.run()

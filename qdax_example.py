import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
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
    import numpy as np
    from scipy.io import wavfile
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc

    from IPython.display import HTML
    from IPython.display import Audio
    import IPython.display as ipd

    import brax
    import jumanji
    import qdax

    import os
    import functools
    import time
    import numpy as np
    from functools import partial
    from typing import List, Tuple, Union, Dict, Optional

    from tqdm.notebook import tqdm

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import flax.linen as nn

    # import brax.v1 as brax
    # import brax.v1.envs
    from brax.v1.envs.env import State

    import qdax
    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
    from qdax.core.emitters.mutation_operators import isoline_variation
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.core.neuroevolution.buffers.buffer import QDTransition
    from qdax.environments import QDEnv
    from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
    from qdax.types import Descriptor
    from qdax.utils.metrics import CSVLogger, default_qd_metrics
    # from qdax.utils.plotting import plot_map_elites_results
    # from qdax.utils.plotting import plot_multidimensional_map_elites_grid

    from IPython.display import Audio
    from IPython.display import clear_output
    import IPython.display as ipd

    from dawdreamer.faust import FaustContext
    from dawdreamer.faust.box import boxFromDSP, boxToSource
    return (
        Audio,
        CSVLogger,
        Descriptor,
        Dict,
        FaustContext,
        HTML,
        List,
        MAPElites,
        MapElitesRepertoire,
        MixingEmitter,
        Optional,
        Path,
        QDEnv,
        QDTransition,
        State,
        Tuple,
        Union,
        animation,
        boxFromDSP,
        boxToSource,
        brax,
        clear_output,
        compute_cvt_centroids,
        createLibContext,
        default_qd_metrics,
        destroyLibContext,
        fbox,
        functools,
        ipd,
        isoline_variation,
        itertools,
        jax,
        jnp,
        jumanji,
        librosa,
        mo,
        mpl,
        nn,
        np,
        optax,
        os,
        partial,
        plt,
        qdax,
        random,
        rc,
        scoring_function,
        time,
        tqdm,
        train_state,
        unfreeze,
        wavfile,
    )


@app.cell
def __(Audio, ipd):
    #@title Constants and Utilities

    SAMPLE_RATE = 44100

    def show_audio(data, autoplay=False):
        if abs(data).max() > 1.:
            data /= abs(data).max()
        
        ipd.display(Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay))
    return SAMPLE_RATE, show_audio


@app.cell
def __():
    #@title QD Training Definitions Fields
    batch_size = 6  #@param {type:"integer"}
    episode_length = 1
    num_iterations = 4000  #@param {type:"integer"}
    log_period = 20
    seed = 42 #@param {type:"integer"}
    iso_sigma = 0.005 #@param {type:"number"}
    line_sigma = 0.01 #@param {type:"number"}
    num_init_cvt_samples = 50000  #@param {type:"integer"}
    num_centroids = 16  #@param {type:"integer"}
    reward_offset = 1e-6 # minimum reward value to make sure qd_score are positive
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
def __(FaustContext, SAMPLE_RATE, boxFromDSP, boxToSource, jax):
    #@title Define Target JAX Model and Ground Truth Parameters

    # Let's make a parametric equalizer using a sequence of 3 filters.
    # https://faustlibraries.grame.fr/libs/filters/#fipeak_eq
    # Each "section" adjusts the gain near a certain frequency, with a bandwidth.

    Section0_Freq = 500 #@param {type:"number"}
    Section0_Gain = -40 #@param {type:"number"}
    Section0_Bandwidth = 100 #@param {type:"number"}

    Section1_Freq = 5000 #@param {type:"number"}
    Section1_Gain = 3 #@param {type:"number"}
    Section1_Bandwidth = 2000 #@param {type:"number"}

    Section2_Freq = 10000 #@param {type:"number"}
    Section2_Gain = -20 #@param {type:"number"}
    Section2_Bandwidth = 4000 #@param {type:"number"}

    Final_Volume = 0 #@param {type:"number"}

    faust_code = f"""
    import("stdfaust.lib");

    // Define the number of sections
    N = 3;

    // Define the parameters for each section.
    freqs = {Section0_Freq}, {Section1_Freq}, {Section2_Freq};
    gains = {Section0_Gain}, {Section1_Gain}, {Section2_Gain};
    bandwidths = {Section0_Bandwidth}, {Section1_Bandwidth}, {Section2_Bandwidth};

    // global volume parameter in decibels
    volume = {Final_Volume};

    section(i) = hgroup("Section %i",
        fi.peak_eq(
            vslider("Gain", gains : ba.selector(i, N), -80., 12., .001), 
            vslider("Freq",  freqs : ba.selector(i, N), 20., 20000., .001),
            vslider("Bandwidth", bandwidths : ba.selector(i, N), 20., 5000., .001)
        )
    );

    // Define a final volume slider.
    vol = _*(vslider("Volume[unit:dB]", volume, -12., 12., .001) : ba.db2linear);

    // Sequence all the sections.
    process = hgroup("Equalizer", seq(i, N, section(i)) : vol);
    """

    module_name = "MyDSP"

    with FaustContext():
        model_source_code = boxToSource(boxFromDSP(faust_code), 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}

    exec(model_source_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]

    model = MyDSP(SAMPLE_RATE)

    env_inference_fn = jax.jit(model.apply, static_argnums=[2])

    # The number of input channels the model takes.
    N_CHANNELS = model.getNumInputs()
    print('N_CHANNELS:', N_CHANNELS)
    return (
        Final_Volume,
        MyDSP,
        N_CHANNELS,
        Section0_Bandwidth,
        Section0_Freq,
        Section0_Gain,
        Section1_Bandwidth,
        Section1_Freq,
        Section1_Gain,
        Section2_Bandwidth,
        Section2_Freq,
        Section2_Gain,
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
    env_inference_fn,
    jax,
    jnp,
    model,
    nn,
    random,
):
    #@title Initialize Batched Model for Training

    N_SECONDS = 1. #@param {type:"number"}
    N_SAMPLES = int(SAMPLE_RATE*N_SECONDS)

    key = random.PRNGKey(0)
    target_variables = model.init({'params': key}, jnp.ones((N_CHANNELS, N_SAMPLES), jnp.float32), N_SAMPLES)
    print('target_variables:', target_variables)
    # Our behavior descriptor length is the number of parameters.
    num_parameters = len(target_variables['params'].keys())
    behavior_descriptor_length = num_parameters
    # behavior_descriptor_length = 2 # todo
    print('behavior_descriptor_length:', behavior_descriptor_length)

    # show the target intermediate parameters which we want our trained model to discover.
    x = -1.+2.*jax.random.uniform(key, (N_CHANNELS, N_SAMPLES))
    y, target_parameters = model.apply(target_variables, x, N_SAMPLES, mutable='intermediates')
    print('target_parameters:', target_parameters['intermediates'])

    random_key, subkey = jax.random.split(key)
    # keys = jax.random.split(subkey, num=batch_size)
    batch_model = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': True})

    # Benchmark the speed of the inference function
    env_inference_fn(target_variables, x, N_SAMPLES)
    return (
        N_SAMPLES,
        N_SECONDS,
        batch_model,
        behavior_descriptor_length,
        key,
        num_parameters,
        random_key,
        subkey,
        target_parameters,
        target_variables,
        x,
        y,
    )


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
    #@title Define a Custom Environment
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
            a_min = [-1. for _ in range(self.behavior_descriptor_length)]
            a_max = [1. for _ in range(self.behavior_descriptor_length)]
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
            x = -1+2.*jax.random.uniform(rng, (N_CHANNELS, N_SAMPLES))

            # use our ground-truth variables to generate audio
            y = env_inference_fn(target_variables, x, N_SAMPLES)

            obs_init = {'x': x, 'y': y}
            reward, done = jnp.zeros(2)
            metrics: Dict = {}
            info_init = {"state_descriptor": obs_init}
            return State(None, obs_init, reward, done, metrics, info_init)

        def step(self, state: State, actions) -> State:
            """Run one timestep of the environment's dynamics."""

            x, y = state.obs['x'], state.obs['y']

            # use actions (learned parameters) to generate audio prediction
            pred = env_inference_fn(actions, x, N_SAMPLES)

            # L1 time-domain loss
            loss = jnp.abs(pred-y).mean()
            
            # keep reward positive so that the QD-score works correctly.
            reward = jnp.maximum(reward_offset, 1.-loss)

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
    key,
    random,
    seed,
    subkey,
):
    #@title Initialize environment, and population params
    # Init environment
    my_env = MyEnv(config=None)

    # Init a random key
    random_key_1 = jax.random.PRNGKey(seed)

    # Create the initial environment states
    random_key_2, subkey_1 = jax.random.split(random_key_1)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(my_env.reset))
    init_states = reset_fn(keys)

    # Randomly initialize the parameters:
    # Pass a fake batch to get init_variables.
    fake_batch = jnp.zeros((batch_size, N_CHANNELS, N_SAMPLES), jnp.float32)
    init_variables = batch_model(SAMPLE_RATE).init(subkey_1, fake_batch, N_SAMPLES)
    # The init_variables start as the ground truth,
    # so we need to randomize them between -1 and 1
    init_variables = jax.tree_util.tree_map(lambda x: random.uniform(key, x.shape, minval=-1, maxval=1, dtype=jnp.float32), init_variables)
    return (
        fake_batch,
        init_states,
        init_variables,
        keys,
        my_env,
        random_key_1,
        random_key_2,
        reset_fn,
        subkey_1,
    )


@app.cell
def __(QDTransition, jax, my_env):
    #@title Define the function to play a step with the policy in the environment
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
    #@title Define the scoring function and the way metrics are computed

    # Prepare the scoring function
    def bd_extraction_fn(data: QDTransition, mask: jnp.ndarray) -> Descriptor:

        matrix = jnp.concatenate(list(data.actions['params'].values()), axis=-1)

        # todo:
        # This is an opportunity to have a behavior space
        # that's smaller than the action space.
        # We could project the matrix from (batch_size, #num params)
        # to (batch_size, behavior_descriptor_length).
        # We don't have to do this right now because they happen to already be the same shape.

        # Suppose `behavior_matrix` has shape (num_parameters, behavior_descriptor_length).
        # Then we could do following projection
        # matrix = matrix @ behavior_matrix

        # We could also use domain knowledge to define new features based on
        # the actions.

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
    #@title Define the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size
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
    random_key_2,
    scoring_fn,
):
    #@title Instantiate and initialize the MAP-Elites algorithm

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key_3 = compute_cvt_centroids(
        num_descriptors=my_env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=-1.,
        maxval=1.,
        random_key=random_key_2,
    )

    # Compute initial repertoire and emitter state

    return centroids, map_elites, random_key_3


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
    random_key_3,
    time,
    tqdm,
):
    repertoire, emitter_state, random_key_4 = map_elites.init(init_variables,
                                                            centroids, random_key_3)
    #@title Launch MAP-Elites iterations
    num_loops = int(num_iterations / log_period)

    csv_logger = CSVLogger(
        "mapelites-logs.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
    )
    all_metrics = {}

    jit_scan = jax.jit(functools.partial(jax.lax.scan, map_elites.scan_update, xs=(), length=log_period))

    pbar = tqdm(range(num_loops))
    for i in pbar:
        start_time = time.time()
        (repertoire, emitter_state, random_key_4,), metrics = jit_scan(
            init=(repertoire, emitter_state, random_key_4),
        )
        timelapse = time.time() - start_time

        # log metrics
        logged_metrics = {"time": timelapse,
                          "loop": 1+i,
                          "iteration": 1 + i*log_period}
        for k, value in metrics.items():
            # take last value
            logged_metrics[k] = value[-1]

            # take all values
            if k in all_metrics.keys():
                all_metrics[k] = jnp.concatenate([all_metrics[k], value])
            else:
                all_metrics[k] = value

        max_fitness = "{:.4f}".format(metrics['max_fitness'].max())
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
        logged_metrics,
        max_fitness,
        metrics,
        num_loops,
        pbar,
        random_key_4,
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
    model,
    np,
    random_key_4,
    repertoire_path,
):
    #@title Loading the Repertoire

    # Init population of policies
    random_key_5, subkey_2= jax.random.split(random_key_4)
    fake_params = model.init(subkey_2, jnp.float32(np.random.random((N_CHANNELS, N_SAMPLES))), N_SAMPLES)

    _, reconstruction_fn = jax.flatten_util.ravel_pytree(fake_params)

    repertoire_final = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)
    return (
        fake_params,
        random_key_5,
        reconstruction_fn,
        repertoire_final,
        subkey_2,
    )


@app.cell(hide_code=True)
def __(
    Dict,
    MapElitesRepertoire,
    Optional,
    Tuple,
    behavior_descriptor_length,
    jnp,
    mpl,
    np,
    plt,
):
    #@title Modified plotting code

    # We use an older version of this file:
    # https://github.com/adaptive-intelligent-robotics/QDax/blob/2fb56198f176f0a9be90baa543b172d4915ba0f9/qdax/utils/plotting.py#L215

    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _get_projection_in_1d(
        integer_coordinates: jnp.ndarray, bases_tuple: Tuple[int, ...]
    ) -> int:
        """Converts an integer vector into a single integer,
        given tuple of bases to consider for conversion.

        This conversion is ensured to be unique, provided that
        for all index i: x[i] < bases_tuple[i].
        The vector and tuple of bases must have the same length.

        For example if x=jnp.array([3, 1, 2]) and the bases are (5, 7, 3).
        then the projection is 3*(7*3) + 1*(3) + 2 = 47.

        Args:
            integer_coordinates: the coordinates of the points (should be integers).
            bases_tuple: the bases to use.

        Returns:
            The projection of the point in 1D (int).
        """
        assert jnp.size(integer_coordinates) == len(
            bases_tuple
        ), "x should have the same size as bases"

        integer_coordinates = integer_coordinates.ravel().tolist()

        # build the conversion
        coordinate = 0
        for x_coord, base in zip(integer_coordinates, bases_tuple):
            coordinate = coordinate * base + x_coord

        return coordinate


    def _get_projection_in_2d(
        integer_coordinates: jnp.ndarray, bases: Tuple[int, ...]
    ) -> Tuple[int, int]:
        """Projects an integer vector into a pair of integers,
        (given tuple of bases to consider for conversion).

        For example if x=jnp.array([3, 1, 2, 5]) and the bases are (5, 2, 3, 7).
        then the projection is obtained by:
        - projecting in 1D the point jnp.array([3, 2]) with the bases (5, 3)
        - projecting in 1D the point jnp.array([1, 5]) with the bases (2, 7)

        Args:
            integer_coordinates: the coordinates of the points (should be integers).
            bases_tuple: the bases to use.

        Returns:
            The projection of the point in 2D (pair of integers).
        """
        integer_coordinates = integer_coordinates.ravel()
        x0 = _get_projection_in_1d(integer_coordinates[::2], bases[::2])
        x1 = _get_projection_in_1d(integer_coordinates[1::2], bases[1::2])
        return x0, x1


    def plot_multidimensional_map_elites_grid(
        repertoire: MapElitesRepertoire,
        minval: jnp.ndarray,
        maxval: jnp.ndarray,
        grid_shape: Tuple[int, ...],
        ax: Optional[plt.Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Tuple[Optional[Figure], Axes]:
        """Plot a visual 2D representation of a multidimensional MAP-Elites repertoire
        (where the dimensionality of descriptors can be greater than 2).

        Args:
            repertoire: the MAP-Elites repertoire to plot.
            minval: minimum values for the descriptors
            maxval: maximum values for the descriptors
            grid_shape: the resolution of the grid.
            ax: a matplotlib axe for the figure to plot. Defaults to None.
            vmin: minimum value for the fitness. Defaults to None.
            vmax: maximum value for the fitness. Defaults to None.

        Raises:
            ValueError: the resolution should be an int or a tuple

        Returns:
            A figure and axes object, corresponding to the visualisation of the
            repertoire.
        """

        descriptors = repertoire.descriptors
        fitnesses = repertoire.fitnesses

        is_grid_empty = fitnesses.ravel() == -jnp.inf
        num_descriptors = descriptors.shape[1]

        if isinstance(grid_shape, tuple):
            assert (
                len(grid_shape) == num_descriptors
            ), "grid_shape should have the same length as num_descriptors"
        else:
            raise ValueError("resolution should be a tuple")

        assert np.size(minval) == num_descriptors or np.size(minval) == 1, (
            f"minval : {minval} should either be of size 1 "
            f"or have the same size as the number of descriptors: {num_descriptors}"
        )
        assert np.size(maxval) == num_descriptors or np.size(maxval) == 1, (
            f"maxval : {maxval} should either be of size 1 "
            f"or have the same size as the number of descriptors: {num_descriptors}"
        )

        non_empty_descriptors = descriptors[~is_grid_empty]
        non_empty_fitnesses = fitnesses[~is_grid_empty]

        # convert the descriptors to integer coordinates, depending on the resolution.
        resolutions_array = jnp.array(grid_shape)
        descriptors_integers = jnp.asarray(
            jnp.floor(
                resolutions_array * (non_empty_descriptors - minval) / (maxval - minval)
            ),
            dtype=jnp.int32,
        )

        # total number of grid cells along each dimension of the grid
        size_grid_x = np.prod(np.array(grid_shape[0::2]))
        # convert to int for the 1d case - if not, value 1.0 is given
        size_grid_y = np.prod(np.array(grid_shape[1::2]), dtype=int)

        # initialise the grid
        grid_2d = jnp.full(
            (size_grid_x, size_grid_y),
            fill_value=jnp.nan,
        )

        # put solutions in the grid according to their projected 2-dimensional coordinates
        for _, (desc, fit) in enumerate(zip(descriptors_integers, non_empty_fitnesses)):
            projection_2d = _get_projection_in_2d(desc, grid_shape)
            if jnp.isnan(grid_2d[projection_2d]) or fit.item() > grid_2d[projection_2d]:
                grid_2d = grid_2d.at[projection_2d].set(fit.item())

        # set plot parameters
        font_size = 12
        params = {
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "text.usetex": False,
            "figure.figsize": [10, 10],
        }

        mpl.rcParams.update(params)

        # create the plot object
        fig = None
        if ax is None:
            fig, ax = plt.subplots(facecolor="white", edgecolor="white")
        ax.set(adjustable="box", aspect="equal")

        my_cmap = cm.viridis

        if vmin is None:
            vmin = float(jnp.min(non_empty_fitnesses))
        if vmax is None:
            vmax = float(jnp.max(non_empty_fitnesses))

        ax.imshow(
            grid_2d.T,
            origin="lower",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap=my_cmap,
        )

        # aesthetic
        ax.set_xlabel("Behavior Dimension 1")
        ax.set_ylabel("Behavior Dimension 2")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
        cbar.ax.tick_params(labelsize=font_size)

        ax.set_title("MAP-Elites Grid")
        ax.set_aspect("equal")

        def _get_ticks_positions(
            total_size_grid_axis: int, step_ticks_on_axis: int
        ) -> jnp.ndarray:
            """
            Get the positions of the ticks on the grid axis.

            Args:
                total_size_grid_axis: total size of the grid axis
                step_ticks_on_axis: step of the ticks

            Returns:
                The positions of the ticks on the plot.
            """
            return np.arange(0, total_size_grid_axis + 1, step_ticks_on_axis) - 0.5

        # Ticks position
        major_ticks_x = _get_ticks_positions(
            size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[2::2]).item()
        )
        minor_ticks_x = _get_ticks_positions(
            size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[4::2]).item()
        )
        major_ticks_y = _get_ticks_positions(
            size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[3::2]).item()
        )
        minor_ticks_y = _get_ticks_positions(
            size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[5::2]).item()
        )

        ax.set_xticks(
            major_ticks_x,
        )
        ax.set_xticks(
            minor_ticks_x,
            minor=True,
        )
        ax.set_yticks(
            major_ticks_y,
        )
        ax.set_yticks(
            minor_ticks_y,
            minor=True,
        )

        # Ticks aesthetics
        ax.tick_params(
            which="minor",
            color="gray",
            labelcolor="gray",
            size=5,
        )
        ax.tick_params(
            which="major",
            labelsize=font_size,
            size=7,
        )

        ax.grid(which="minor", alpha=1.0, color="#000000", linewidth=0.5)
        ax.grid(which="major", alpha=1.0, color="#000000", linewidth=2.5)

        ax.set_xticklabels(
            [
                f"{x:.2}"
                for x in jnp.around(
                    jnp.linspace(minval[0], maxval[0], num=len(major_ticks_x)), decimals=2
                )
            ]
        )
        ax.set_yticklabels(
            [
                f"{y:.2}"
                for y in jnp.around(
                    jnp.linspace(minval[1], maxval[1], num=len(major_ticks_y)), decimals=2
                )
            ]
        )

        return fig, ax

    def plot_map_elites_results(
        env_steps: jnp.ndarray,
        metrics: Dict,
        repertoire: MapElitesRepertoire,
        min_bd: jnp.ndarray,
        max_bd: jnp.ndarray,
        grid_shape: int=2
    ):

        """Plots three usual QD metrics, namely the coverage, the maximum fitness
        and the QD-score, along the number of environment steps. This function also
        plots a visualisation of the final map elites grid obtained. It ensures that
        those plots are aligned together to give a simple and efficient visualisation
        of an optimization process.
        Args:
            env_steps: the array containing the number of steps done in the environment.
            metrics: a dictionary containing metrics from the optimizatoin process.
            repertoire: the final repertoire obtained.
            min_bd: the mimimal possible values for the bd.
            max_bd: the maximal possible values for the bd.
        Returns:
            A figure and axes with the plots of the metrics and visualisation of the grid.
        """
        # Customize matplotlib params
        font_size = 16
        params = {
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "text.usetex": False,
            "axes.titlepad": 10,
        }

        mpl.rcParams.update(params)

        # Visualize the training evolution and final repertoire
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))

        # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

        axes[0].plot(env_steps, metrics["coverage"])
        axes[0].set_xlabel("Environment steps")
        axes[0].set_ylabel("Coverage in %")
        axes[0].set_title("Coverage evolution during training")
        axes[0].set_aspect(0.95 / axes[0].get_data_ratio(), adjustable="box")

        axes[1].plot(env_steps, metrics["max_fitness"])
        axes[1].set_xlabel("Environment steps")
        axes[1].set_ylabel("Maximum fitness")
        axes[1].set_title("Maximum fitness evolution during training")
        axes[1].set_aspect(0.95 / axes[1].get_data_ratio(), adjustable="box")

        axes[2].plot(env_steps, metrics["qd_score"])
        axes[2].set_xlabel("Environment steps")
        axes[2].set_ylabel("QD Score")
        axes[2].set_title("QD Score evolution during training")
        axes[2].set_aspect(0.95 / axes[2].get_data_ratio(), adjustable="box")

        minval = jnp.full((behavior_descriptor_length,), min_bd)
        maxval = jnp.full((behavior_descriptor_length,),  max_bd)
        grid_shape = tuple([2]*behavior_descriptor_length)
        _, axes = plot_multidimensional_map_elites_grid(repertoire,
                                                        minval=minval,
                                                        maxval=maxval,
                                                        grid_shape=grid_shape,
                                                        ax=axes[3])

        return fig, axes
    return (
        Axes,
        Figure,
        Normalize,
        cm,
        make_axes_locatable,
        plot_map_elites_results,
        plot_multidimensional_map_elites_grid,
    )


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
    #@title Plotting

    # Create the x-axis array
    env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    print(repertoire.centroids.shape)
    # Create the plots and the grid
    fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics,
                                        repertoire=repertoire,
                                        min_bd=-1., max_bd=1.,
                                        grid_shape=2
                                        )

    fig

    # Note that our MAP-Elites Grid will look sparse when the `num_centroids` hyperparameter is small.
    return axes, env_steps, fig


@app.cell
def __(jnp, repertoire):
    #@title Show the Best Parameters
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_fitness = jnp.max(repertoire.fitnesses)
    best_bd = repertoire.descriptors[best_idx]

    print(
        f"Best fitness in the repertoire: {best_fitness:.4f}\n",
        f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
        f"Index in the repertoire of this individual: {best_idx}\n"
    )
    return best_bd, best_fitness, best_idx


if __name__ == "__main__":
    app.run()

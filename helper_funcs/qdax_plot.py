#@title Modified plotting code

# We use an older version of this file:
# https://github.com/adaptive-intelligent-robotics/QDax/blob/2fb56198f176f0a9be90baa543b172d4915ba0f9/qdax/utils/plotting.py#L215

from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax
import jax.numpy as jnp
from typing import List, Tuple, Union, Dict, Optional
import qdax
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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
    grid_shape: int=2,
    behavior_descriptor_length: int = 3,
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

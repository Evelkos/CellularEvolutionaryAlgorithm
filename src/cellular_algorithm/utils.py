import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


def plot_population_on_the_surface(
    function,
    boundaries=(-100, 100),
    points=30,
    ax=None,
    population_coordinates=None,
):
    X, Y, Z = compute_surface(function, boundaries, points)

    # Create an empty plot
    ax_in = ax
    ax = ax if ax is not None else plt.axes(projection="3d")

    plot_surface(
        ax=ax,
        title=function.__name__,
        surface=(X, Y, Z),
    )
    # If population has been given, plot population on the surface.
    if population_coordinates:
        plot_population(ax, population_coordinates)

    # If `ax` has not been given, display the plot.
    if ax_in is None:
        plt.show()

    return ax


def plot_surface(
    ax,
    title,
    surface,
):
    """
    Creates a surface plot of a function.

    Args:
        function (function): The objective function to be called at each point.
        boundaries (tuple[tuple[float, float]]: describes range of possible solutions
            eg. ((0, 10), (100, 200), (3, 15)) =>
            0 < x < 10, 100 < y < 200, 3 < z < 15
        domain (num, num): The inclusive (min, max) domain for each dimension.
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed.
        ax (matplotlib axes): Optional axes to use (must have projection='3d').
            Note, if specified plt.show() will not be called.
    """
    # create points^2 tuples of (x,y) and populate z
    X, Y, Z = surface

    ax.plot_surface(X, Y, Z, cmap="gist_ncar", edgecolor="none", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax


def compute_surface(
    function,
    boundaries=(-100, 100),
    points=30,
):
    """
    Creates a surface plot of a function.

    Args:
        function (function): The objective function to be called at each point.
        boundaries (tuple[tuple[float, float]]: describes range of possible solutions
            eg. ((0, 10), (100, 200), (3, 15)) =>
            0 < x < 10, 100 < y < 200, 3 < z < 15
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed.
    """
    # create points^2 tuples of (x,y) and populate z
    xys = np.array(
        [np.linspace(min(boundary), max(boundary), points) for boundary in boundaries]
    )
    xys = np.transpose([np.tile(xys[0], len(xys[1])), np.repeat(xys[1], len(xys[0]))])
    zs = np.zeros(points * points)

    if len(boundaries) > 2:
        # concatenate remaining zeros
        tail = np.zeros(len(boundaries) - 2)
        for i in range(0, xys.shape[0]):
            zs[i] = function(np.concatenate([xys[i], tail]))
    else:
        for i in range(0, xys.shape[0]):
            zs[i] = function(xys[i])

    X = xys[:, 0].reshape((points, points))
    Y = xys[:, 1].reshape((points, points))
    Z = zs.reshape((points, points))

    return X, Y, Z


def plot_population(ax, population_coordinates):
    X = [x for x, y, z in population_coordinates]
    Y = [y for x, y, z in population_coordinates]
    Z = [z for x, y, z in population_coordinates]
    points_on_the_plot = ax.scatter(X, Y, Z, s=1, c="black")

    return points_on_the_plot


def __generate_frame(ax, evolution, surface, title, camera):
    plot_surface(ax=ax, title=title, surface=surface)
    population_coordinates = evolution.get_population_coordinates()
    plot_population(ax, population_coordinates)
    camera.snap()


def record(evolution, points=20, iteration_step=10, filename=None):
    """Record evolution.

    Displays surface and population.

    Arguments:
        evolution: evolution to run
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed
        iteration_step: number of iterations to get next snaps
        filename: path to the file where movie will be saved.
            eg. .mp4 or .gif.
            WARNING: .mp4 file requires `ffmpeg` installed!
            If None, movie will be displayed.

    """
    function = evolution.function
    boundaries = evolution.boundaries
    surface = compute_surface(function, boundaries, points)

    ax = plt.axes(projection="3d")
    camera = Camera(ax.figure)

    # Add first frame (evolution has not started yet)
    __generate_frame(ax, evolution, surface, function.__name__, camera)

    # Add single frame aftear each iteration
    for iteration in evolution.step_run():
        if iteration % iteration_step == 0:
            __generate_frame(ax, evolution, surface, function.__name__, camera)

    animation = camera.animate()

    # Display or save image
    if filename is None:
        plt.show()
    else:
        animation.save(filename)

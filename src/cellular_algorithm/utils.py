import statistics

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
    """Plot population and the surface.

    Arguments:
        function: function that will be used to generate surface
            It should be the same as evolution's function
        boundaries: what part of the surface should be displayed
            It should be the same as evolution's boundaries
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed
        ax: ax. If None, plot will be displayed to the User
        population_coordinates: population that should be displayed on the surface
            If None, only surface will be displayed.

    Return:
        ax

    """
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

    Return:
        ax

    """
    # create points^2 tuples of (x,y) and populate z
    X, Y, Z = surface

    ax.plot_surface(X, Y, Z, cmap="gist_ncar", edgecolor="none", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax


def compute_surface(function, boundaries=(-100, 100), points=30):
    """Compute surface of given function.

    Args:
        function (function): The objective function to be called at each point.
        boundaries (tuple[tuple[float, float]]: describes range of possible solutions
            eg. ((0, 10), (100, 200), (3, 15)) =>
            0 < x < 10, 100 < y < 200, 3 < z < 15
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed.

    Return:
        Lists X (x coordinates), Y (y coordinates), Z (z coordinates) that defines
        surface.

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
    """Plot 3D population.

    Arguments:
        ax: ax
        population_coordinates: coordinates of the population

    """
    X = [x for x, y, z in population_coordinates]
    Y = [y for x, y, z in population_coordinates]
    Z = [z for x, y, z in population_coordinates]
    points_on_the_plot = ax.scatter(X, Y, Z, s=1, c="black")

    return points_on_the_plot


def __generate_frame_3d(ax, population_coordinates, surface, title, camera):
    """Generate 3D frame. Take snap.

    Arguments:
        ax: ax
        population_coordinates: population's coordinates from given iteration
        surface: surface returned by compute_surface()
        title: title of a plot
        camera: celluloid's Camera object
    """
    plot_surface(ax=ax, title=title, surface=surface)
    plot_population(ax, population_coordinates)
    camera.snap()


def __generate_frame_2d(ax, population_coordinates, surface, title, camera, levels=40):
    """Generate 2D frame. Take snap.

    Arguments:
        ax: ax
        population_coordinates: population's coordinates from given iteration
        surface: surface returned by compute_surface()
        title: this argument will be ignored
        camera: celluloid's Camera object
        levels: "depth" of the image

    """
    x, y, z, *_ = surface

    # Plot surface (only colors)
    ax.contourf(x, y, z, cmap="RdBu_r", levels=levels)

    # Plot population 2d
    X = [x for x, y, *z in population_coordinates]
    Y = [y for x, y, *z in population_coordinates]
    ax.scatter(X, Y, s=2, c="black")

    camera.snap()


def record(
    population_trace,
    evolution,
    points=20,
    iteration_step=10,
    filename=None,
    mode="2D",
    ax=None,
    display=False,
):
    """Record evolution in 3d.

    Displays surface and population.

    Arguments:
        population_trace: list of populations' coordinates in each iteration
        evolution: evolution object that has been used to generate `population_trace`
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed
        iteration_step: number of iterations to get next snap
        filename: path to the file where movie will be saved.
            eg. .mp4 or .gif.
            WARNING: .mp4 file requires `ffmpeg` installed!
            If None, movie will be displayed.
        mode: "2D" or "3D"
        ax: ax
        display: if animation should be displayed

    Returns:
        list of populations' coordinates in each iteration

    """
    if mode not in {"2D", "3D"}:
        raise ValueError("Only 2D and 3D modes are allowed")

    # Prepare empty plot, choose frame generator
    if mode == "3D":
        ax = ax if ax is not None else plt.axes(projection="3d")
        generate_frame = __generate_frame_3d
    elif mode == "2D":
        ax = ax if ax is not None else plt.subplots()[1]
        generate_frame = __generate_frame_2d

    title = evolution.function.__name__
    ax.set_title(title)

    camera = Camera(ax.figure)
    surface = compute_surface(evolution.function, evolution.boundaries, points)

    # Record population after given number of `iteration_step`s
    for iteration, population_coordinates in enumerate(population_trace):
        if iteration % iteration_step == 0:
            generate_frame(ax, population_coordinates, surface, title, camera)

    # Display or save image
    animation = camera.animate()

    if display:
        plt.show()
    if filename is not None:
        animation.save(filename)


def summary(population_trace):
    """Compute min, max and mean value for each iteration.

    Arguments:
        population_trace: list of populations' coordinates in each iteration

    Return:
        dictionary with keys:
            - max_fitness - list of max fitnesses in each iteration
            - min_fitness - list of min fitnesses in each iteration
            - mean_fitness - list of mean fitnesses in each iteration

    """
    max_fitness = []
    min_fitness = []
    mean_fitness = []

    for iteration in population_trace:
        max_fitness.append(max(iteration, key=lambda x: x[-1])[-1])
        min_fitness.append(min(iteration, key=lambda x: x[-1])[-1])
        mean_fitness.append(statistics.mean(individual[-1] for individual in iteration))

    return {
        "max_fitness": max_fitness,
        "min_fitness": min_fitness,
        "mean_fitness": mean_fitness,
    }


def summary_plots(summary_dict, ax=None):
    """Plot max, min and mean fitness of the population in each iteration.

    Arguments:
        summary_dict: dictionary returned by summary()
        ax: ax or None

    """
    ax__original = ax
    if ax is None:
        fig, ax = plt.subplots()

    x = range(len(summary_dict["max_fitness"]))

    ax.plot(x, summary_dict["max_fitness"], label="max fitness")
    ax.plot(x, summary_dict["min_fitness"], label="min fitness")
    ax.plot(x, summary_dict["mean_fitness"], label="mean fitness")

    ax.set_title("Population distribution")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fitness")

    ax.legend(loc="upper right")

    if ax__original is None:
        plt.show()
    else:
        return ax


def population_fitness_plot(population_trace, ax=None):
    """Plot fitnesses of all individuals in each iteration.

    Arguments:
        population_trace: trace of the population returned by evolution.run(save_trace=True)
        ax: ax or None

    """
    ax__original = ax
    if ax is None:
        fig, ax = plt.subplots()

    for iteration, population in enumerate(population_trace):
        x = [iteration for _ in range(len(population))]
        y = [individual[-1] for individual in population]
        ax.scatter(x, y, s=1, c="black")

    ax.set_title("Population fitness distribution")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fitness")

    if ax__original is None:
        plt.show()
    else:
        return ax

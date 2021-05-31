import random

import numpy as np

import cec2017.basic as basic
import cec2017.functions as functions
import cec2017.simple as simple
import cec2017.utils as utils
from cellular_algorithm import (
    CompactNeighborhood,
    Evolution,
    GaussianMutation,
    LinearNeighborhood,
    SinglePointCrossover,
    TournamentSelection,
    UniformCrossover,
    RankSelection,
)


def my_function(x):
    sm = 0
    for i in range(len(x)):
        sm += (x[i] - 4) ** 2
    return sm


def get_minimum(individual_1, individual_2):
    if individual_1.fitness < individual_2.fitness:
        return individual_1
    return individual_2


if __name__ == "__main__":
    # grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # result = CompactNeighborhood.get_neighbours(grid.shape, (1, 0), 1)
    # print(result)
    random.seed(42)

    neighbourhood = CompactNeighborhood(distance=1)
    # selection = TournamentSelection(tournament_size=2, parents_num=2)
    selection = RankSelection(fraction=2/9, min_parents=2)
    crossover = UniformCrossover
    mutation = GaussianMutation(scale=1)

    boundaries = ((0, 10), (10, 20))
    shape = (10, 10)
    evolution = Evolution(
        neighbourhood=neighbourhood,
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        boundaries=boundaries,
        function=my_function,
        maximize=False,
        shape=shape,
        grid=None,
        mutation_probability=1,
        iterations=100,
    )
    print(evolution.run())
    # print(evolution.grid)

    # neighbourhood = LinearNeighborhood(distance=1)
    # grid = Grid2D(shape=[10, 10], neighbourhood=neighbourhood)

    # # Accepted dimensions are 2, 10, 20, 30, 50 or 100
    # # (f11 - f20 and f29 - f30 not defined for D = 2)
    # D = 2

    # # evaluate a specific function a few times
    # f = functions.f5
    # for i in range(0, 10):
    #     x = np.random.uniform(low=-100, high=100, size=D)
    #     y = f(x)
    #     print("%s(%.1f,%.1f) = %.2f" % (f.__name__, x[0], x[1], y))

    # # make a surface plot of f27
    # utils.surface_plot(functions.f1, points=120)

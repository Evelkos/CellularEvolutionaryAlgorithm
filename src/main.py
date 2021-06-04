import random

import numpy as np

import cec2017.basic as basic
import cec2017.functions as functions
import cec2017.simple as simple
import cec2017.utils as utils
from cellular_algorithm import (
    CellularEvolutionaryAlgorithm,
    CompactNeighborhood,
    Evolution,
    EvolutionaryAlgorithm,
    GaussianMutation,
    LinearNeighborhood,
    RankSelection,
    SinglePointCrossover,
    TournamentSelection,
    UniformCrossover,
    plot_population_on_the_surface,
    record,
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
    neighbourhood = CompactNeighborhood(distance=1)
    selection = TournamentSelection(tournament_size=2, parents_num=2)
    crossover = UniformCrossover
    mutation = GaussianMutation(scale=1)

    boundaries = ((0, 10), (10, 20))
    shape = (10, 10)
    # evolution = CellularEvolutionaryAlgorithm(
    #     neighbourhood=neighbourhood,
    #     crossover=crossover,
    #     mutation=mutation,
    #     selection=selection,
    #     boundaries=boundaries,
    #     function=my_function,
    #     maximize=False,
    #     population_shape=shape,
    #     mutation_probability=1,
    #     iterations=10,
    # )
    # print(evolution.run())

    selection = TournamentSelection(tournament_size=3, parents_num=2)
    evolution = EvolutionaryAlgorithm(
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        boundaries=boundaries,
        function=my_function,
        maximize=False,
        mutation_probability=1,
        iterations=100,
        population_shape=(1, 100),
    )

    # plot_population_on_the_surface(
    #     my_function,
    #     points=120,
    #     population_coordinates=population_coordinates,
    #     boundaries=boundaries,
    # )

    record(evolution)

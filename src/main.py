import random

import numpy as np

import cec2017.basic as basic
import cec2017.functions as functions
import cec2017.simple as simple
import cec2017.utils as utils
from cellular_algorithm import (CellularEvolutionaryAlgorithm,
                                CompactNeighborhood, Evolution,
                                EvolutionaryAlgorithm, GaussianMutation,
                                LinearNeighborhood, RankSelection,
                                SinglePointCrossover, TournamentSelection,
                                UniformCrossover,
                                plot_population_on_the_surface,
                                population_fitness_plot, record, summary,
                                summary_plots)


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
    neighbourhood = CompactNeighborhood(distance=2)
    selection = TournamentSelection(tournament_size=2, parents_num=2)
    crossover = UniformCrossover
    mutation = GaussianMutation(scale=6)

    boundaries = ((-100, 100), (-100, 100))
    shape = (10, 10)
    # evolution = CellularEvolutionaryAlgorithm(
    #     neighbourhood=neighbourhood,
    #     crossover=crossover,
    #     mutation=mutation,
    #     selection=selection,
    #     boundaries=boundaries,
    #     function=functions.f6,
    #     maximize=False,
    #     population_shape=shape,
    #     mutation_probability=1,
    #     iterations=200,
    # )
    evolution = EvolutionaryAlgorithm(
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        boundaries=boundaries,
        function=functions.f5,
        maximize=False,
        mutation_probability=1,
        iterations=200,
        population_shape=(1, 100),
    )

    evolution_trace = evolution.run(save_trace=True)
    
    my_summary = summary(evolution_trace)
    summary_plots(my_summary)

    # population_fitness_plot(evolution_trace)

    # record(evolution_trace, evolution, points=20, iteration_step=10, filename=None)

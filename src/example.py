import numpy as np

import cec2017.basic as basic
import cec2017.functions as functions
import cec2017.simple as simple
import cec2017.utils as utils
from cellular_algorithm import (CellularEvolutionaryAlgorithm,
                                CompactNeighborhood, Evolution,
                                EvolutionaryAlgorithm, GaussianMutation,
                                LinearNeighborhood, RankSelection,
                                RankSuccession, RouletteWheelSelection,
                                RouletteWheelSuccession, SinglePointCrossover,
                                TournamentSelection, TournamentSuccession,
                                UniformCrossover,
                                plot_population_on_the_surface,
                                population_fitness_plot, record, summary,
                                summary_plots)


def my_function(x):
    """Simple function that can be minimized."""
    sm = 0
    for i in range(len(x)):
        sm += (x[i] - 4) ** 2
    return sm


if __name__ == "__main__":
    neighbourhood = CompactNeighborhood(distance=2)
    selection = TournamentSelection(tournament_size=2)
    # selection = RouletteWheelSelection()
    crossover = UniformCrossover
    mutation = GaussianMutation(scale=6)
    # succession = RouletteWheelSuccession()
    succession = TournamentSuccession(tournament_size=2)

    # Use my_function() or one of cec2017 functions. Eg. functions.f3
    function = functions.f5

    # Boundaries of the function.
    boundaries = ((-100, 100), (-100, 100))
    # Shape of the grid
    shape = (10, 10)

    evolution = CellularEvolutionaryAlgorithm(
        neighbourhood=neighbourhood,
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        succession=succession,
        boundaries=boundaries,
        function=function,
        maximize=False,
        population_shape=shape,
        mutation_probability=1,
        iterations=200,
    )
    # evolution = EvolutionaryAlgorithm(
    #     crossover=crossover,
    #     mutation=mutation,
    #     selection=selection,
    #     succession=succession,
    #     boundaries=boundaries,
    #     function=function,
    #     maximize=False,
    #     mutation_probability=1,
    #     iterations=200,
    #     population_shape=(1, 100),
    # )
    # Run evolution, save its trace.
    evolution_trace = evolution.run(save_trace=True)

    # Plot summary with min, max and mean fitnesses.
    my_summary = summary(evolution_trace)
    summary_plots(my_summary, filename=None, display=True)

    # Plot fitnesses of the individuals.
    population_fitness_plot(evolution_trace, filename=None, display=True)

    # Use filename with `.gif` extension if ffmpeg has not been installed.
    # Record evolution in 2D or 3D.
    record(
        evolution_trace,
        evolution,
        points=20,
        iteration_step=10,
        mode="3D",
        filename=None,
        display=True,
    )

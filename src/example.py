import argparse
import sys

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


def get_arguments(command_args):
    """If EvolutionaryAlgorithm or CellularEvolutionaryAlgorithm should be used."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea",
        action="store_true",
        help=(
            "Add this flag if classic evolution should be used instead of cellular "
            "evolution"
        ),
    )
    args = parser.parse_args(command_args)
    return args.ea


if __name__ == "__main__":
    ea = get_arguments(sys.argv[1:])

    neighbourhood = CompactNeighborhood(distance=2)
    selection = TournamentSelection(tournament_size=2)
    crossover = UniformCrossover
    mutation = GaussianMutation(scale=6)
    succession = TournamentSuccession(tournament_size=2)
    # Use my_function() or one of cec2017 functions. Eg. functions.f3
    function = functions.f6
    # Boundaries of the function.
    boundaries = ((-100, 100), (-100, 100))
    # Shape of the grid
    shape = (10, 10)
    iterations = 1000

    if ea:
        evolution = EvolutionaryAlgorithm(
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            succession=succession,
            boundaries=boundaries,
            function=function,
            maximize=False,
            mutation_probability=0.3,
            iterations=iterations,
            population_shape=shape,
        )
    else:
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
            mutation_probability=0.3,
            iterations=iterations,
        )

    # Run evolution, save its trace.
    evolution_trace = evolution.run(save_trace=True)

    # Plot summary with min, max and mean fitnesses.
    my_summary = summary(evolution_trace)
    summary_plots(my_summary, filename=None, display=True)
    # Plot fitnesses of the individuals.
    population_fitness_plot(evolution_trace, filename=None, display=False)
    # Use filename with `.gif` extension if ffmpeg has not been installed.
    # Record evolution in 2D or 3D.
    record(
        evolution_trace,
        evolution,
        points=20,
        iteration_step=int(iterations / 20),  # Generate 20 frames
        mode="2D",
        filename=None,
        display=True,
    )
    record(
        evolution_trace,
        evolution,
        points=20,
        iteration_step=int(iterations / 20),  # Generate 20 frames
        mode="3D",
        filename=None,
        display=True,
    )

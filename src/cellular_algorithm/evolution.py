import random
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from cellular_algorithm import Grid


class Evolution(ABC):
    def __init__(
        self,
        crossover,
        mutation,
        selection,
        succession,
        boundaries,
        function,
        maximize=True,
        mutation_probability=1,
        iterations=100,
        parents_num=2,
        population_shape=(1, 100),
        population=None,
    ):
        """
        Arguments:
            crossover: crossover that will be used to create new individuals
            mutation: type of mutation that will be used to modify new individuals
            selection: type of mutation
            succession: type of succession
            boundaries: describes range of possible solutions
                eg. ((0, 10), (100, 200), (3, 15)) =>
                0 < x < 10, 100 < y < 200, 3 < z < 15
            function: function that will be optimized
            maximize: if function should be maximized (if not, it will be minimized)
            mutation_probability: probability of mutation
            iterations: number of iterations
            parents_num: number of parents that will be used to create new individual
            population_shape: shape of the grid
                - (1, population_num) - for classic evolution
                - (n_1, ..., n_x) - for cellular evolution
            population - population

        """

        self.crossover = crossover
        self.selection = selection
        self.succession = succession
        self.mutation = mutation

        self.boundaries = boundaries
        self.function = function
        self.maximize = maximize

        self.mutation_probability = mutation_probability
        self.iterations = iterations
        self.parents_num = parents_num

        if not population and not population_shape:
            raise ValueError("You need to specify `grid` or `shape` to create it.")

        if not population:
            population = Grid(population_shape)
            population.generate_individuals(self.boundaries, self.function)

        self.population = population
        self.population_shape = self.population.grid.shape
        # Create tmp grid that will be used in each iteration
        self.offsprings = Grid(self.population_shape)

        self.best_solution = None
        self.best_solution_position = None

    def get_best(self, individuals):
        if self.maximize:
            return max(*individuals, key=lambda x: x.fitness)
        else:
            return min(*individuals, key=lambda x: x.fitness)

    def update_best_solution(self, individual, position):
        if (
            not self.best_solution
            or self.get_best([self.best_solution, individual]) is individual
        ):
            self.best_solution = individual
            self.best_solution_position = position

    def select_parents(self, individuals):
        """Selection.

        Arguments:
            individuals: population or part of the population that we want to select
                parents from.

        """
        return self.selection.select(individuals, self.maximize, num=self.parents_num)

    def recombine(self, parents):
        """Crossover.

        Recombine selected parents to create new individual.

        Arguments:
            parents: list of parents to recombine. There should be exactly 2 parents

        Return:
            newly created individual

        """
        return self.crossover.recombine(*parents)

    def normalize_coordinates(self, individual):
        """Make sure that individual's coordinates meet boundaries."""
        result = []
        for boundary, coordinate in zip(self.boundaries, individual.coordinates):
            low, high = boundary
            coordinate = coordinate if coordinate >= low else float(low)
            coordinate = coordinate if coordinate <= high else float(high)
            result.append(coordinate)

        individual.coordinates = np.array(result)
        return individual

    def mutate(self, new_individual):
        """Mutation."""
        return self.mutation.mutate(new_individual)

    def get_population_coordinates(self):
        return [
            (*individual.coordinates, individual.fitness)
            for _, individual in self.population.iterate_individuals()
        ]

    @abstractmethod
    def choose_next_population(self):
        """Succession."""
        ...

    @abstractmethod
    def run_single_iteration(self):
        ...

    def run(self, save_trace=False):
        """Run evolution."""
        population_trace = None

        if save_trace:
            population_trace = [self.get_population_coordinates()]

        for iteration in tqdm(range(self.iterations)):
            self.run_single_iteration()
            if save_trace:
                population_trace.append(self.get_population_coordinates())

        return population_trace


class EvolutionaryAlgorithm(Evolution):
    def __init__(self, crossover_probability=1, *args, **kwargs):
        """
        Arguments:
            crossover_probability: probability of the crossover

        """
        super(EvolutionaryAlgorithm, self).__init__(*args, **kwargs)
        self.crossover_probability = crossover_probability

    def choose_next_population(self):
        """Succession."""
        current_population = self.population.get_all_individuals()
        next_population = self.succession.select(
            current_population,
            self.offsprings.get_all_individuals(),
            self.maximize,
            len(current_population),
        )
        self.population.grid = np.reshape(next_population, self.population_shape)

    def run_single_iteration(self):
        for grid_position, individual in self.population.iterate_individuals():
            # Selection and crossover
            if random.uniform(0, 1) < self.crossover_probability:
                parents = self.select_parents(self.population.get_all_individuals())
                new_individual = self.recombine(parents)
            else:
                new_individual = self.population.get_random_individual()
            # Mutation
            if random.uniform(0, 1) < self.mutation_probability:
                new_individual = self.mutate(new_individual)
            # Normalization and fitness computation
            new_individual = self.normalize_coordinates(new_individual)
            new_individual.fitness = self.function(new_individual.coordinates)

            self.offsprings.set_individual(new_individual, grid_position)
            self.update_best_solution(new_individual, grid_position)

        # Succession
        self.choose_next_population()


class CellularEvolutionaryAlgorithm(Evolution):
    def __init__(self, neighbourhood, *args, **kwargs):
        """
        Arguments:
            neighbourhood: describes type of neighbourhood

        """
        super(CellularEvolutionaryAlgorithm, self).__init__(*args, **kwargs)
        self.neighbourhood = neighbourhood

    def select_parents(self, grid_position):
        """Selection.

        Select individual's neighbours that will be used to create new individual.

        Arguments:
            grid_position: individual's position on the grid.

        Return:
            List of neighbours that will be used to create new individual.

        """
        positions = self.neighbourhood.get_neighbours(
            self.population_shape, grid_position
        )
        neighbours = self.population.get_individuals(positions)
        return super().select_parents(neighbours)

    def choose_next_population(self):
        """Succession.

        For each cell choose better individual.

        Arguments:
            offsprings: grid with newly create individuals

        """
        for individual_info, offspring_info in zip(
            self.population.iterate_individuals(),
            self.offsprings.iterate_individuals(),
        ):
            position, individual = individual_info
            _, offspring = offspring_info

            result = self.succession.select(
                np.array([individual]), np.array([offspring]), self.maximize, 1
            )
            self.population.set_individual(result[0], position)

    def run_single_iteration(self):
        for grid_position, individual in self.population.iterate_individuals():
            # Selection
            parents = self.select_parents(grid_position)
            # Crossover
            new_individual = self.recombine(parents)
            # Mutation
            if random.uniform(0, 1) < self.mutation_probability:
                new_individual = self.mutate(new_individual)
            # Normalization and fitness computation
            new_individual = self.normalize_coordinates(new_individual)
            new_individual.fitness = self.function(new_individual.coordinates)

            self.offsprings.set_individual(new_individual, grid_position)
            self.update_best_solution(new_individual, grid_position)

        # Succession.
        self.choose_next_population()

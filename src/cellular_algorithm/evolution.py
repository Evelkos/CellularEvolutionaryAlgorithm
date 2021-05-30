import random

from tqdm import tqdm

from cellular_algorithm import Grid


class Evolution:
    def __init__(
        self,
        neighbourhood,
        crossover,
        mutation,
        selection,
        boundaries,
        function,
        maximize=True,
        shape=None,
        grid=None,
        mutation_probability=1,
        iterations=100,
    ):
        """
        Arguments:
            neighbourhood: describes type of neighbourhood
            crossover: crossover that will be used to create new individuals
            mutation: type of mutation that will be used to modify new individuals
            selection: type of mutation
            boundaries: describes range of possible solutions
                eg. ((0, 10), (100, 200), (3, 15)) =>
                0 < x < 10, 100 < y < 200, 3 < z < 15
            function: function that will be optimized
            maximize: if function should be maximized (if not, it will be minimized)
            shape: shape of the grid.
                eg. (10, 20, 30) => 6000 individuals
            grid: Grid with existing individuals
            mutation_probability: probability of mutation
            iterations: number of iterations

        """
        if not grid and not shape:
            raise ValueError(
                "You need to specify `grid` of individuals or `shape` to create it."
            )

        if not grid:
            grid = Grid(shape, neighbourhood)
            grid.generate_individuals(boundaries, function)

        self.grid = grid

        self.neighbourhood = neighbourhood
        self.crossover = crossover
        self.selection = selection
        self.mutation = mutation

        self.boundaries = boundaries
        self.function = function
        self.maximize = maximize

        self.grid_shape = self.grid.grid.shape
        self.mutation_probability = mutation_probability
        self.iterations = iterations

        self.best_solution = None
        self.best_solution_position = None

    def get_better(self, individual_1, individual_2):
        if self.maximize:
            return max([individual_1, individual_2], key=lambda x: x.fitness)
        else:
            return min([individual_1, individual_2], key=lambda x: x.fitness)

    def update_best_solution(self, individual, position):
        if (
            not self.best_solution
            or self.get_better(self.best_solution, individual) is individual
        ):
            self.best_solution = individual
            self.best_solution_position = position

    def select_parents(self, grid_position):
        """Selection.

        Select individual's neighbours that will be used to create new individual.

        Arguments:
            grid_position: individual's position on the grid.

        Return:
            List of neighbours that will be used to create new individual.

        """
        neighbours_positions = self.neighbourhood.get_neighbours(
            grid_shape=self.grid_shape, idx=grid_position
        )
        # TODO: optimize
        neighbours = [self.grid.get_individual(pos) for pos in neighbours_positions]
        return self.selection.select(neighbours)

    def recombine(self, parents):
        """Crossover.

        Recombine individual's neighbours to create new individual.

        Arguments:
            parents: list of parents to recombine. There should be exactly
                2 parents

        Return:
            newly created individual

        """
        return self.crossover.recombine(*parents)

    def mutate(self, new_individual):
        """Mutation."""
        return self.mutation.mutate(new_individual)

    def succession(self, individual, new_individual):
        return self.get_better(individual, new_individual)

    def run(self):
        tmp_grid = Grid(self.grid_shape, self.neighbourhood)

        for i in tqdm(range(self.iterations)):
            for grid_position, individual in self.grid.get_individuals():
                # Selection
                parents = self.select_parents(grid_position)
                # Crossover
                new_individual = self.recombine(parents)
                # Mutation
                if random.uniform(0, 1) < self.mutation_probability:
                    new_individual = self.mutate(new_individual)
                # Fitness computation
                new_individual.fitness = self.function(new_individual.coordinates)
                # Succession
                result_individual = self.succession(individual, new_individual)

                tmp_grid.set_individual(result_individual, grid_position)
                self.update_best_solution(result_individual, grid_position)

            self.grid = tmp_grid

        return self.best_solution

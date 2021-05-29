import random

from cellular_algorithm import Grid


class Evolution:
    def __init__(
        self,
        neighbourhood,
        boundaries,
        function,
        shape=None,
        grid=None,
        distance=1,
        parents_num=2,
        with_replacement=True,
    ):
        """
        Arguments:
            neighbourhood: describes type of neighbourhood
            boundaries: describes range of possible solutions
                eg. ((0, 10), (100, 200), (3, 15)) =>
                0 < x < 10, 100 < y < 200, 3 < z < 15
            function: function that will be optimized
            shape: shape of the grid.
                eg. (10, 20, 30) => 6000 individuals
            grid: Grid with existing individuals
            distance: describes max distance between individual and its neighbours
            parents_num: number of neighbours used to create new individual
            with_replacement: if items should be drawn with replacement

        """
        if not grid and not shape:
            raise ValueError(
                "You need to specify `grid` of individuals or `shape` to create it."
            )

        if grid:
            self.grid = grid
        else:
            self.grid = Grid(shape, neighbourhood)
            self.grid.generate_individuals(boundaries)

        self.neighbourhood = neighbourhood
        self.boundaries = boundaries
        self.function = function
        self.grid_shape = self.grid.shape
        self.distance = distance
        self.parents_num = parents_num
        self.with_replacement = with_replacement

    def selection(self, grid_position):
        """Select individual's neighbours that will be used to create new individual.

        Arguments:
            grid_position: individual's position on the grid.

        Return:
            Positions of individuals' neighbours that will be used to produce new
            individual.

        """
        neighbours_positions = self.neighbourhood.get_neighbours(
            grid_shape=self.grid_shape, idx=grid_position, distance=self.distance
        )
        if self.with_replacement:
            return random.choices(neighbours_positions, k=self.parents_num)
        else:
            return random.sample(neighbours_positions, self.parents_num)

    def crossover(self):
        ...

    def mutation(self):
        ...

    def succession(self):
        ...

    def run(self):
        for grid_position, individual in self.grid.get_individuals():
            neighbours_positions = self.selection(grid_position)
            print(f"{grid_position}: {neighbours_positions}")

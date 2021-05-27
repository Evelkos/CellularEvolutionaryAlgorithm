from abc import ABC, abstractmethod

import numpy as np

from .individual import Individual


class Grid(ABC):
    ...


class Grid2D(Grid):
    def __init__(self, shape, neighbourhood):
        """
        Arguments:
            shape: describes Grid's shape
            neighbourhood: type of neighbourhood

        """
        self.grid = np.empty(shape, dtype=Individual)
        self.neighbourhood = neighbourhood

    def get_individuals(self):
        for position, individual in np.ndenumerate(self.grid):
            yield position, individual

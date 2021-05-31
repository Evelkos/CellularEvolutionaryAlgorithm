import random
from abc import ABC, abstractmethod

import numpy as np

from cellular_algorithm import Individual


class Crossover(ABC):
    @abstractmethod
    def recombine(parent_1, parent_2):
        """Recombine parents to create new Individual.

        Both parents need to have list of coordinates of the same length.

        Arguments:
            parent_1: first parent
            parent_2: second parent

        Returns:
            Newly created Individual

        """
        ...


class SinglePointCrossover(Crossover):
    @abstractmethod
    def recombine(parent_1, parent_2):
        """Use single point to recombine parent's coordinates and create new Individual.

        Both parents need to have list of coordinates of the same length

        Arguments:
            parent_1: first parent
            parent_2: second parent

        Returns:
            Newly created Individual

        """
        point = random.randint(0, len(parent_1.coordinates))
        return Individual(
            coordinates=np.concatenate(
                [parent_1.coordinates[:point], parent_2.coordinates[point:]]
            ),
            fitness=None,
        )


class UniformCrossover(Crossover):
    @abstractmethod
    def recombine(parent_1, parent_2):
        """Use single point to recombine parent's coordinates and create new Individual.

        Both parents need to have list of coordinates of the same length

        Arguments:
            parent_1: first parent
            parent_2: second parent

        Returns:
            Newly created Individual

        """
        coordinates = []
        for idx in range(len(parent_1.coordinates)):
            if random.uniform(0, 1) < 0.5:
                coordinates.append(parent_1.coordinates[idx])
            else:
                coordinates.append(parent_2.coordinates[idx])

        return Individual(coordinates=np.array(coordinates), fitness=None)

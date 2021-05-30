from abc import ABC, abstractmethod

import numpy as np


class Mutation(ABC):
    @abstractmethod
    def mutate(individual):
        ...


class GaussianMutation(Mutation):
    def __init__(self, scale):
        self.scale = scale
        self.loc = 0

    def mutate(self, individual):
        """Add noise to the individual's coordinates.

        Arguments:
            individual: individual that will be modified

        Return:
            modified individual

        """
        noise = np.random.normal(
            loc=self.loc, scale=self.scale, size=len(individual.coordinates)
        )
        individual.coordinates += noise
        return individual

from abc import ABC, abstractmethod

import numpy as np

from cellular_algorithm import (RankSelection, RouletteWheelSelection,
                                TournamentSelection)


class Succession(ABC):
    @abstractmethod
    def select(self, individuals, offsprings, maximize, num):
        ...


class TournamentSuccession(TournamentSelection, Succession):
    """Select individuals using TournamentSelection.

    Select individuals from `population` and `offsprings`.

    """

    def select(self, population, offsprings, maximize, num):
        individuals = np.concatenate((population, offsprings), axis=None)
        return super().select(individuals=individuals, maximize=maximize, num=num)


class RouletteWheelSuccession(RouletteWheelSelection, Succession):
    """Select individuals using RouletteWheelSelection.

    Select individuals from `population` and `offsprings`.

    """

    def select(self, population, offsprings, maximize, num):
        individuals = np.concatenate((population, offsprings), axis=None)
        return super().select(individuals=individuals, maximize=maximize, num=num)


class RankSuccession(RankSelection, Succession):
    """Select individuals using RankSelection.

    Select individuals from `population` and `offsprings`.

    """

    def select(self, population, offsprings, maximize, num):
        individuals = np.concatenate((population, offsprings), axis=None)
        return super().select(individuals=individuals, maximize=maximize, num=num)


class BasicSuccession(Succession):
    """Select individuals from `offsprings` only."""

    def select(self, population, offsprings, maximize, num):
        return RankSuccession().select(offsprings, maximize, num)

import math
import random
from abc import ABC, abstractmethod


class Selection(ABC):
    @abstractmethod
    def select(self, individuals, maximize, num):
        ...


class TournamentSelection(Selection):
    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def get_winner(self, tournament, maximize):
        if maximize:
            return max(tournament, key=lambda individual: individual.fitness)
        else:
            return min(tournament, key=lambda individual: individual.fitness)

    def select(self, individuals, maximize, num):
        """
        Arguments:
            individuals: list of the individuals we choose from
            maximize: if fitness should be maximized or not (minimized)
            num: number of individuals that should be returned

        """
        parents = []
        for _ in range(num):
            tournament = random.choices(individuals, k=self.tournament_size)
            winner = self.get_winner(tournament, maximize)
            parents.append(winner)
        return parents


class RouletteWheelSelection(Selection):
    ...


class RankSelection(Selection):
    def __init__(self, fraction=None):
        """
        Arguments:
            fraction: float  (0, 1)

        """
        if fraction:
            assert fraction > 0 and fraction < 1, "Fraction needs to be > 0 and < 1"
        self.fraction = fraction

    def select(self, individuals, maximize, num):
        """
        Arguments:
            individuals: list of the individuals we choose from
            maximize: if fitness should be maximized or not (minimized)
            num: number of individuals that should be returned. If `fraction` has been
                given, it will be used instead of `num`

        """
        # maximize = True => descending
        individuals = sorted(individuals, key=lambda x: x.fitness, reverse=maximize)
        if self.fraction:
            individuals_num = math.ceil(len(individuals) * self.fraction)
        else:
            individuals_num = num

        return individuals[:individuals_num]

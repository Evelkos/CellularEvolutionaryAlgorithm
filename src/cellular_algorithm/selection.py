import random
from abc import ABC, abstractmethod


class Selection(ABC):
    @abstractmethod
    def select(self, individuals, num, params):
        ...


class TournamentSelection(Selection):
    def __init__(self, tournament_size, parents_num, maximize=True):
        self.tournament_size = tournament_size
        self.parents_num = parents_num
        self.get_winner = max if maximize else min

    def select(self, individuals):
        """
        Arguments:
            individuals: list of the individuals we choose from

        """
        parents = []
        for _ in range(self.parents_num):
            tournament = random.choices(individuals, k=self.tournament_size)
            winner = self.get_winner(
                tournament, key=lambda individual: individual.fitness
            )
            parents.append(winner)
        return parents


class RouletteWheelSelection(Selection):
    ...


class RankSelection(Selection):
    ...

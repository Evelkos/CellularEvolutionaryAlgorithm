class Individual:
    def __init__(self, coordinates, fitness=None):
        self.coordinates = coordinates
        self.fitness = fitness

    def __repr__(self):
        return str(self.coordinates)

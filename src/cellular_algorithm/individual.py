class Individual:
    def __init__(self, coordinates, fitness):
        self.coordinates = coordinates
        self.fitness = fitness

    def __repr__(self):
        return f"Individual ({self.fitness}): {self.coordinates}"

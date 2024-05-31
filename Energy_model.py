import Simulator.unit_values as UNIT

class Model:
    def __init__(self):
        self.datamovement = 0
        self.computation = 0
        self.total = 0

    def get_energy(self, counter, size, unit, category):
        sub_energy = counter*size*unit*UNIT.NUM_BITS
        if "data" in category:
            self.datamovement += sub_energy
        elif "computaion" in category:
            self.computation += sub_energy
        self.total += sub_energy
        return sub_energy
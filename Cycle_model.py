class Model:
    def __init__(self):
        self.datamovement = 0
        self.computation = 0
        self.total = 0

    def get_cycle(self, counter, size, unit, category):
        sub_cycle = counter*size*unit
        if "data" in category:
            self.datamovement += sub_cycle
        elif "computaion" in category:
            self.computation += sub_cycle
        self.total += sub_cycle
        return sub_cycle
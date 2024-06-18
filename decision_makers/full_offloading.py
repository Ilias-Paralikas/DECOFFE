from .decision_maker_base import DescisionMakerBase
import numpy as np
class FullOffloading(DescisionMakerBase):
    def __init__(self, number_of_actions, *args, **kwargs):
        self.n = number_of_actions-1


    def choose_action(self, *args, **kwargs):
            self.choice = np.random.randint(2)  # Random choice between 0, 1, and 2
            if self.choice == 0:
                return np.random.randint(1, self.n)  # Random choice between 1 and n-1
            elif self.choice == 1:
                return self.n
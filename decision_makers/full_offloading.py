from .decision_maker_base import DescisionMakerBase
import random
class FullOffloading(DescisionMakerBase):
    def __init__(self, number_of_actions, *args, **kwargs):
        self.n = number_of_actions-1


    def choose_action(self, *args, **kwargs):
            self.choice = random.randint(1, 2)  # Random choice between 0, 1, and 2
            if self.choice == 1:
                return random.randint(1, self.n-1)  # Random choice between 1 and n-1
            elif self.choice == 2:
                return self.n
from .decision_maker_base import DescisionMakerBase

class RoundRobin(DescisionMakerBase):
    def __init__(self, number_of_actions, *args, **kwargs):
        self.n = number_of_actions-1
        self.current = 0  # Start from 1 for round-robin
        self.choice = 0

    def choose_action(self, *args, **kwargs):
        self.choice = (self.choice + 1) % 3  # Round-robin choice between 0, 1, and 2
        if self.choice == 0:
            return 0
        elif self.choice == 1:
            result = self.current
            self.current = (self.current % self.n) + 1  # Round-robin from 1 to n
            return result
        elif self.choice == 2:
            return self.n 
   
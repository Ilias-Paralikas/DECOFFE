from .flbase import FlBase
import torch
class SimpleAverage(FlBase):
    def __init__(self,attributes=['Q_eval_network','Q_target_network'],*args, **kwargs):
        self.attributes = attributes
        pass
    
    def average_weights(self,agents,*args,**kwargs):
        def average_one_network(agents,attribute):
            sum_weights = None
            for agent in agents:
                state_dict = getattr(agent, attribute).state_dict()
                if sum_weights is None:
                    sum_weights = {name: torch.zeros_like(param) for name, param in state_dict.items()}
                for name, param in state_dict.items():
                    sum_weights[name] += param
            average_weights = {name: param / len(agents) for name, param in sum_weights.items()}

            for agent in agents:
                getattr(agent, attribute).load_state_dict(average_weights)
        for attribute in self.attributes:
            average_one_network(agents,attribute)

  

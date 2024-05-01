from .flbase import FlBase
import torch
class SimpleAverage(FlBase):
    def __init__(self,averaging_frequency,attributes=['Q_eval_network','Q_target_network'],*args, **kwargs):
        super().__init__(averaging_frequency,attributes,*args, **kwargs)
    
    def average_weights(self,agents,*args,**kwargs):
        sum_weights = None
        for agent in agents:
            state_dict =agent.Q_eval_network.state_dict()
            if sum_weights is None:
                sum_weights = {name: torch.zeros_like(param) for name, param in state_dict.items()}
            for name, param in state_dict.items():
                sum_weights[name] += param
        average_weights = {name: param / len(agents) for name, param in sum_weights.items()}

        for agent in agents:
            agent.Q_eval_network.load_state_dict(average_weights)
                
        

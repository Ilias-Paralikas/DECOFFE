import torch
class FlBase():
    def __init__(self,averaging_frequency=0,*args, **kwargs):
        self.averaging_frequency = averaging_frequency
        self.counter = 0
        
    def count_epochs(self):
        if self.averaging_frequency:
            self.counter +=1
            self.counter = self.counter % self.averaging_frequency  
            if self.counter == self.averaging_frequency :
                return True
        return False
                
    def average_weights(self,agents,*args,**kwargs):
        pass
        
    def compare_state_dicts(self,agent):
        base_state_dict = agent[0].Q_eval_network.state_dict()
        for model in agent[1:]:
            other_state_dict = model.Q_eval_network.state_dict()
            for (k1, v1), (k2, v2) in zip(base_state_dict.items(), other_state_dict.items()):
                if k1 != k2 or not torch.equal(v1, v2):
                    return False
        return True
        
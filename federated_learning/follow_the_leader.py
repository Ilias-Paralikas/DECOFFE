from .flbase import FlBase
import torch
import numpy as np
class FollowTheLeader(FlBase):
    def __init__(self,averaging_frequency,attributes=['Q_eval_network','Q_target_network'],*args, **kwargs):
        super().__init__(averaging_frequency,attributes,*args, **kwargs)
    
    def average_weights(self, agents,agent_average_scores, *args, **kwargs):
        # Find the agent with the highest score
        leader_index =  np.argmax(agent_average_scores)
        # Copy the weights of the agent with the highest score to all other agents
        leader_weights = agents[leader_index].Q_eval_network.state_dict()
        for agent in agents:
            agent.Q_eval_network.load_state_dict(leader_weights)
        
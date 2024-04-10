from .decision_maker_base import DescisionMakerBase
import math
import numpy as np 

class ULOOF(DescisionMakerBase):
    def __init__(self, 
                 id,
                number_of_actions,
                hyperparameters,
                 *args, **kwargs):
        self.id = id
        self.number_of_actions = number_of_actions
        self.timeout_delay =  hyperparameters['timeout_delay']
        self.task_computational_density = hyperparameters['task_computational_density']
        self.servers_private_queues_computational_capacity = hyperparameters['servers_private_queues_computational_capacities'][0] *hyperparameters['delta_duration']
        self.servers_public_queues_computational_capacity = hyperparameters['servers_public_queues_computational_capacities'][0] *hyperparameters['delta_duration']
        self.horizontal_transmission_capacities =  hyperparameters['transmission_capacities'][0][0]*hyperparameters['delta_duration']
        self.vertical_transmission_capacities =  hyperparameters['transmission_capacities'][0][-1]*hyperparameters['delta_duration']
        self.cloud_computational_capacity = hyperparameters['cloud_computational_capacity']*hyperparameters['delta_duration']
        
        self.action_delay = np.zeros(self.number_of_actions)
        self.local_process_per_time_period = self.servers_private_queues_computational_capacity / self.task_computational_density
        
    def choose_action(self, local_observations,active_queues):
        def get_local_time(bitarrive):
            time_to_process_task = math.ceil(bitarrive/self.local_process_per_time_period)
            return time_to_process_task
        def get_horizontal_transmission_time(bitarrive):
            return  math.ceil(bitarrive/self.horizontal_transmission_capacities)
        def get_vertical_transmission_time(bitarrive):
            return  math.ceil(bitarrive/self.vertical_transmission_capacities)
         
        def get_offloading_time(bitarrive,active_queues):
            process_per_time_period = (self.servers_public_queues_computational_capacity
                                       / (self.task_computational_density *(1+active_queues)))
            time_to_process_task = math.ceil(bitarrive/process_per_time_period)
            return time_to_process_task
        
        
        bitarrive = local_observations[0]
        processing_queue_waiting_time    = local_observations[1]
        transmission_queue_waiting_time  = local_observations[2]
        horizontal_transmission_time = get_horizontal_transmission_time(bitarrive)
        vertical_transmission_time = get_vertical_transmission_time(bitarrive)
        
        
        self.action_delay[0] =processing_queue_waiting_time+get_local_time(bitarrive)
        for i in range(1,self.number_of_actions-1):
            self.action_delay[i] =(transmission_queue_waiting_time+
                                   horizontal_transmission_time+ 
                                   get_offloading_time(bitarrive,active_queues[i-1]))
            
        self.action_delay[-1] = (transmission_queue_waiting_time+
                                 vertical_transmission_time+
                                 get_offloading_time(bitarrive,active_queues[-1])                                )
        
        
    
        action = np.argmin(self.action_delay)
        return action
        
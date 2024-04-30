import numpy as np
from .task import Task
from .queues import PublicQueueManager
 
 
class Cloud:
    def __init__(self,
                 number_of_servers ,
                 computational_capacity):
        self.number_of_servers=  number_of_servers
        self.computational_capacity = computational_capacity
        
        self.current_time=0
        public_queue_hash_map = [i for i in range(self.number_of_servers)]
        reward_hash_map = [i for i in range(self.number_of_servers)]
        self.public_queues_manager = PublicQueueManager(id=number_of_servers,
                                                        number_of_servers=number_of_servers,
                                                        number_of_supporting_servers=number_of_servers,
                                                        computational_capacity=self.computational_capacity,
                                                        public_queue_hash_map=public_queue_hash_map,
                                                        reward_hash_map=reward_hash_map)
        

    def reset(self):
        self.current_time=0
        self.public_queues_manager.reset()
            

    
    def step(self,tasks=[]):
        self.public_queues_manager.add_tasks(tasks)
        rewards = self.public_queues_manager.step()
        return rewards
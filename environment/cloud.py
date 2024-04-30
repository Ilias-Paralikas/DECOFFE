import numpy as np
from .task import Task
from .queues import PublicQueue
 
class Cloud:
    def __init__(self,
                 number_of_servers ,
                 computational_capacity):
        self.number_of_servers=  number_of_servers
        self.computational_capacity = computational_capacity
        
        self.current_time=0
        self.public_queues = [PublicQueue() for _ in range(self.number_of_servers)]
        

    def reset(self):
        self.current_time=0
        for q in self.public_queues:
            q.reset()


    def get_active_queues(self):
        active_queues =0
        total_priority =0
        for q in self.public_queues:
            if not q.is_empty():
                active_queues +=1
                total_priority += q.current_task.task_priority
        return active_queues,total_priority
    
    
    def step(self,tasks=[]):
        rewards =np.zeros(self.number_of_servers)
        for task in tasks:
            origin_server_id = task.origin_server_id
            self.public_queues[origin_server_id].add_task(task)
    
        active_queues,total_priority= self.get_active_queues()
        
        if active_queues!=0:
            distributed_computational_capacity = self.computational_capacity/total_priority
        else:
            distributed_computational_capacity = 0
        for i,q in enumerate(self.public_queues):
            rewards[i] = q.process(distributed_computational_capacity)
        
        self.current_time +=1
        return rewards
import queue
from .task import Task
import math
import numpy as np

class TaskQueue:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.current_time=0
        self.queue_length = 0
        self.queue = queue.Queue()
        self.current_task = Task()
    
    def is_empty(self):
        return  self.current_task.is_empty and self.queue.empty()
    
    def add_task(self,task):
        if self.is_empty():
            self.current_task = task
        else:
            self.queue.put(task)

    def current_task_is_timed_out(self):
        return self.current_time >= self.current_task.arrival_time + self.current_task.timeout_delay


    def get_first_non_empty_element(self):
        self.current_time +=1
        reward = 0
        if self.current_task.is_empty:
            if self.queue.empty():
                return reward
            else:
                self.current_task = self.queue.get()
        while self.current_task_is_timed_out():
            reward -=  self.current_task.drop_task()
            self.queue_length -= self.current_task.remain
            if self.queue.empty():
                return reward
            else:
                self.current_task = self.queue.get()
        return reward
    


class ProcessingQueue(TaskQueue):
    def __init__(self,computational_capacity):
        super().__init__()
        self.computational_capacity = computational_capacity
        self.waiting_time =0
    def reset(self):
        super().reset()
        self.waiting_time =0        

    def update_waiting_time(self,task):
        process_per_time_period = self.computational_capacity / task.task_computational_density
        time_to_process_task = math.ceil(task.size/process_per_time_period)
        timeout_time = max(0,task.timeout_delay-self.waiting_time)
        self.waiting_time += min(timeout_time,time_to_process_task)
         
    def add_task(self,task):
        super().add_task(task)
        self.update_waiting_time(task)
        
    def process(self):
        if self.waiting_time>0:
            self.waiting_time -=1
        reward = self.get_first_non_empty_element()
        if self.current_task.is_empty:
            return reward
        self.current_task.remain -= self.computational_capacity / self.current_task.task_computational_density
        if self.current_task.remain <=0 :   
            self.current_task.finish_task()
            reward -=  self.current_time-self.current_task.arrival_time+1
        
        return reward
    


class OffloadingQueue(TaskQueue):
    def __init__(self):   
        super().__init__()
        self.waiting_time =0
    def reset(self):
        super().reset()
        self.waiting_time =0
        
    
    def update_waiting_time(self, task,offloading_capacity):
        time_to_transmit_task =  math.ceil(task.size/ offloading_capacity)
        timeout_time = max(0,task.timeout_delay- self.waiting_time)
        self.waiting_time += min(timeout_time,time_to_transmit_task)
        
    def add_task(self,task):
        super().add_task(task)
        self.update_waiting_time(task,task.offloading_capacity)
    
    def transmit(self):
        if self.waiting_time>0:
            self.waiting_time -=1
        transmited_task = None
        reward = self.get_first_non_empty_element()
        if self.current_task.is_empty:
            return transmited_task,reward

        self.current_task.remain -=  self.current_task.offloading_capacity
        if self.current_task.remain <=0 :  
            transmited_task = self.current_task.get_offloaded_task()
        return transmited_task,reward
 
class PublicQueue(TaskQueue):
    
    def add_task (self,task):
        super().add_task(task)
        self.queue_length += task.size

        
    def process(self,computational_capacity):
        reward = self.get_first_non_empty_element()
        if self.current_task.is_empty:
            return reward
        computational_capacity =  computational_capacity  * self.current_task.task_priority 
        task_processed = computational_capacity / self.current_task.task_computational_density
        self.current_task.remain -=task_processed
        self.queue_length -= task_processed
        if self.current_task.remain <=0 :   
            self.queue_length -= self.current_task.remain
            self.current_task.finish_task()
            reward -=  self.current_time-self.current_task.arrival_time+1
        return reward
                    


class PublicQueueManager():
    def __init__(self,
                 id,
                 number_of_servers,
                 number_of_supporting_servers,
                 computational_capacity,
                 public_queue_hash_map,
                 reward_hash_map):
        self.id = id
        self.number_of_servers = number_of_servers
        self.number_of_supporting_servers = number_of_supporting_servers
        self.computational_capacity = computational_capacity
        self.public_queues = [PublicQueue() for _ in range(self.number_of_supporting_servers)]
        self.public_queue_hash_map= public_queue_hash_map
        self.reward_hash_map =reward_hash_map
    
    def reset(self):
        for q in self.public_queues:
            q.reset()
    
    def get_public_queue_server_length(self,server_id):
        public_queue_id = self.public_queue_hash_map[server_id]
        return self.public_queues[public_queue_id].queue_length
    
    def get_active_queues(self):
        active_queues =0
        total_priority =0
        for q in self.public_queues:
            if not q.is_empty():
                active_queues +=1
                total_priority += q.current_task.task_priority
        return active_queues,total_priority
        
    def add_tasks(self,recieved_tasks=[]):
        for task in recieved_tasks:
            assert task.target_server_id == self.id
            origin_server_id = task.origin_server_id
            destination_public_queue = self.public_queue_hash_map[origin_server_id]
            self.public_queues[destination_public_queue].add_task(task)
    
    def step(self):
        rewards =np.zeros(self.number_of_servers)
        active_queues,total_priority= self.get_active_queues()
        if active_queues!=0:
            distributed_computational_capacity = self.computational_capacity/total_priority
        else:
            distributed_computational_capacity = 0
        for i,q in enumerate(self.public_queues):
            reward_server = self.reward_hash_map[i]
            rewards[reward_server] = q.process(distributed_computational_capacity)
        return rewards
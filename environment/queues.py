import queue
from .task import Task
import math


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
                    

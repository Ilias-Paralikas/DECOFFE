

class Task:
    def __init__(self,
                size=None,
                arrival_time=None,
                task_computational_density =None,
                timeout_delay=None,
                task_drop_penalty_multiplier=None,
                origin_server_id = None,
                target_server_id = None):
        if size :
            self.is_empty = False
            self.size=size
            self.remain=size
    
            self.arrival_time=arrival_time
            self.timeout_delay=timeout_delay
            self.task_computational_density = task_computational_density
            self.task_drop_penalty_multiplier= task_drop_penalty_multiplier
            self.task_drop_penalty  = task_drop_penalty_multiplier*timeout_delay
         
            self.origin_server_id=origin_server_id
            self.target_server_id=target_server_id
        
        else:
            self.is_empty =True
                
    def drop_task(self):
        self.is_empty = True
        return self.task_drop_penalty
        
    def finish_task(self):
        self.is_empty = True


    def wrap_task_to_offload(self,origin_server_id,target_server_id,offloading_capacity):
        self.origin_server_id= origin_server_id
        self.target_server_id= target_server_id
        self.offloading_capacity=offloading_capacity
    
    def get_offloaded_task(self):
        self.is_empty = True
        return Task(size=self.size,
                    arrival_time=self.arrival_time,
                    timeout_delay=self.timeout_delay,
                    task_computational_density=self.task_computational_density,
                    task_drop_penalty_multiplier= self.task_drop_penalty_multiplier,
                    origin_server_id= self.origin_server_id,
                    target_server_id= self.target_server_id)
        
      
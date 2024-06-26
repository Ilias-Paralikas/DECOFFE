import numpy as np
from .queues import OffloadingQueue,ProcessingQueue,PublicQueueManager



class Server:
    def __init__(self,
                 id,
                 number_of_servers ,
                 server_priority,
                 private_queue_computational_capacity,
                 public_queues_computational_capacity,
                 offloading_queue_transmision_capacities,
                 number_of_clouds=1):
        self.id=id
        self.server_priority = server_priority
        self.number_of_servers=  number_of_servers
        self.private_queue_computational_capacity = private_queue_computational_capacity
        self.public_queues_computational_capacity = public_queues_computational_capacity
        self.offloading_queue_transmision_capacities = offloading_queue_transmision_capacities
        self.public_queue_hash_map = {}
        public_queue_counter =0
        for i in range(number_of_servers):
            if i != id:
                self.public_queue_hash_map[i] = public_queue_counter
                public_queue_counter +=1
                
        self.transmisiom_queue_hash_map= {}
        transmisiom_queue_counter = 0
        for i in range(number_of_servers+number_of_clouds):
             if i != id:
                self.transmisiom_queue_hash_map[i] = transmisiom_queue_counter
                transmisiom_queue_counter +=1

    
        self.action_hash_map = [i for i in range(self.number_of_servers+1) if i != self.id]
        self.action_hash_map.insert(0, self.id)
        
        self.reward_hash_map = [i for i in range(self.number_of_servers+1) if i != self.id]
    
     

                
        
        assert len(offloading_queue_transmision_capacities) == number_of_servers
        
        self.processing_queue = ProcessingQueue(self.private_queue_computational_capacity)
        self.public_queue_manager = PublicQueueManager(id= id,
                                                       number_of_servers=number_of_servers,
                                                       number_of_supporting_servers=number_of_servers-1,
                                                       computational_capacity=public_queues_computational_capacity,
                                                       public_queue_hash_map=self.public_queue_hash_map,
                                                       reward_hash_map=self.reward_hash_map)
        
        self.offloading_queue = OffloadingQueue()
        self.current_time=0

    def reset(self):
        self.current_time=0
        self.processing_queue.reset()
        self.public_queue_manager.reset()
        self.offloading_queue.reset()

    def get_waiting_times(self):
        return  self.processing_queue.waiting_time,self.offloading_queue.waiting_time
    
    
    
    def step(self,action=None,local_task=None):
        offloaded_rewards =np.zeros(self.number_of_servers)

        if local_task:
            local_task.arrival_time=self.current_time
            if action ==0:
                self.processing_queue.add_task(local_task)               
            else:
                target_server_id = self.action_hash_map[action]
                transmission_capacity_index = self.transmisiom_queue_hash_map[target_server_id]
                transmission_capacity = self.offloading_queue_transmision_capacities[transmission_capacity_index]
                local_task.wrap_task_to_offload(self.id,target_server_id,transmission_capacity)
                self.offloading_queue.add_task(local_task)
                
        
        offloaded_rewards += self.public_queue_manager.step()
        
        pocessing_reward= self.processing_queue.process()
        transmited_task, transmission_reward = self.offloading_queue.transmit()
        offloaded_rewards[self.id] += transmission_reward

        local_rewards = np.zeros(self.number_of_servers)
        local_rewards[self.id] = pocessing_reward
        self.current_time +=1
        return transmited_task,local_rewards, offloaded_rewards

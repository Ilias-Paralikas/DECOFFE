
import numpy as np
from .task import Task
from .servers import Server
from .cloud import Cloud


class Environment:
    def __init__(self,
                 servers_private_queues_computational_capacities,
                 servers_public_queues_computational_capacities,
                 transmission_capacities,
                 cloud_computational_capacity,
                 episode_time,
                 timeout_delay,
                 max_bit_arrive ,
                 min_bit_arrive ,
                 task_arrive_probability,
                 delta_duration,
                 task_drop_penalty_multiplier,
                 task_computational_density,
                 server_priorities,
                 local_variables = 3,
                 number_of_clouds =1):
        
        
        self.number_of_servers = len(servers_private_queues_computational_capacities)
        self.number_of_clouds  = number_of_clouds
        self.local_variables = local_variables
        assert len(servers_private_queues_computational_capacities) == len(servers_public_queues_computational_capacities)
        assert transmission_capacities.shape[0] == len(servers_private_queues_computational_capacities)
        assert transmission_capacities.shape[1] == len(servers_private_queues_computational_capacities)-1 + self.number_of_clouds
        

        self.state_dimensions = local_variables +(self.number_of_servers -1) +self.number_of_clouds # they reviece their
        self.lstm_variables = self.number_of_clouds +(self.number_of_servers  -1) # everyone predicts the public queues of the other servers
        self.number_of_actions = self.number_of_servers  + self.number_of_clouds
        
        self.servers_private_queue_computational_capacities = servers_private_queues_computational_capacities * delta_duration
        self.servers_public_queues_computational_capacities = servers_public_queues_computational_capacities * delta_duration
        self.transmission_capacities = transmission_capacities * delta_duration
        self.cloud_computational_capacity = cloud_computational_capacity * delta_duration
         
        self.episode_time= episode_time
        self.timeout_delay= timeout_delay
        self.min_bit_arrive = min_bit_arrive
        self.max_bit_arrive= max_bit_arrive
        self.task_arrive_probability=  task_arrive_probability
        self.episode_time_end = episode_time+timeout_delay
        self.task_drop_penalty_multiplier = task_drop_penalty_multiplier
        self.task_computational_density = task_computational_density
        

        self.servers =[Server(id=i,
                              server_priority = server_priorities[i],
                              number_of_servers = self.number_of_servers,
                              private_queue_computational_capacity = self.servers_private_queue_computational_capacities[i],
                              public_queues_computational_capacity = self.servers_public_queues_computational_capacities[i],
                              offloading_queue_transmision_capacities = self.transmission_capacities[i])
                       for i in range(self.number_of_servers)]
        
        self.cloud  = Cloud(
            number_of_servers=self.number_of_servers,
            computational_capacity=  self.cloud_computational_capacity )
        
        self.current_time = 0
        self.reset_tasks_to_be_transmited()

    
    def reset(self):
        self.current_time = 0

        for s in self.servers:
            s.reset()
        self.cloud.reset()
        self.reset_tasks_to_be_transmited()
        
        
        self.bitarrive = np.random.uniform(self.min_bit_arrive, self.max_bit_arrive, size= self.number_of_servers) 
        self.bitarrive = self.bitarrive * (np.random.uniform(0, 1, size=[self.number_of_servers])< self.task_arrive_probability)

        local_observations = np.zeros((self.number_of_servers,
                                self.state_dimensions)) 
        active_queues  =[0  for _ in range(self.number_of_servers +self.number_of_clouds)]         
                                                                
        for i,bit in enumerate(self.bitarrive):
            local_observations[i][0]=bit
        
        observations = (local_observations,active_queues)

        return observations

    def get_agent_variables(self):
            return self.state_dimensions,self.lstm_variables,self.number_of_actions
    
    def reset_tasks_to_be_transmited(self):
        self.horizontal_transmissions ={}
        for server in range(self.number_of_servers):
            self.horizontal_transmissions[server] =[]
        self.cloud_transmissions =[]
    
    
    def step(self,actions):
        if self.current_time >=self.episode_time_end:
            done = True
        else:
            done = False
        
        rewards = self.cloud.step(self.cloud_transmissions)
        offloaded_rewards  = rewards.copy()
        for target_server,tasks in self.horizontal_transmissions.items():
            self.servers[target_server].public_queue_manager.add_tasks(tasks)
        
        
        self.reset_tasks_to_be_transmited()
        
        
        for server in self.servers:
            task_size = self.bitarrive[server.id] 
            if task_size  ==0:
                transmited_task,local_rewards,offloaded_server_rewards = server.step()
            else:
                action = actions[server.id]
                task = Task(size=task_size,
                            task_priority = server.server_priority,
                            timeout_delay=self.timeout_delay,
                            task_drop_penalty_multiplier=self.task_drop_penalty_multiplier,
                            task_computational_density=self.task_computational_density) 
                transmited_task,local_rewards,offloaded_server_rewards = server.step(action,task)
            offloaded_rewards += offloaded_server_rewards
            rewards += local_rewards
            rewards += offloaded_server_rewards
            if transmited_task:
                # 0 to n-1 are the servers n is the cloud
                if transmited_task.target_server_id ==self.number_of_servers:
                    self.cloud_transmissions.append(transmited_task)
                else:
                    target_server = transmited_task.target_server_id
                    self.horizontal_transmissions[target_server].append(transmited_task)
               

        old_bitarrive = self.bitarrive

        if self.current_time < self.episode_time: 
            self.bitarrive = np.random.uniform(self.min_bit_arrive, self.max_bit_arrive, size= self.number_of_servers) 
            self.bitarrive = self.bitarrive * (np.random.uniform(0, 1, size=[self.number_of_servers])< self.task_arrive_probability)
        else:
            self.bitarrive = np.zeros(self.number_of_servers)
        

        local_observations = np.zeros((self.number_of_servers,
                                self.state_dimensions))
            
        
        for server in self.servers:
            local_observations[server.id][0] = self.bitarrive[server.id]
            processing_queue_waiting_time,transmission_queue_waiting_time = server.get_waiting_times()
            local_observations[server.id][1] = processing_queue_waiting_time
            local_observations[server.id][2] = transmission_queue_waiting_time
            
            for s in self.servers:
                if s.id  != server.id:
                    length_index = server.public_queue_hash_map[s.id]
                    local_observations[server.id][self.local_variables+length_index] = s.public_queue_manager.get_public_queue_server_length(server.id)
            
            local_observations[server.id][-1] = self.cloud.public_queues_manager.get_public_queue_server_length(server.id)
            
        self.current_time +=1
        rewards   = rewards/(self.task_drop_penalty_multiplier*self.timeout_delay)
        offloaded_rewards = offloaded_rewards/(self.task_drop_penalty_multiplier*self.timeout_delay)
            
            
        active_queues =[s.public_queue_manager.get_active_queues()[0] for s in self.servers]           
        active_queues.append(self.cloud.public_queues_manager.get_active_queues()[0])     
        observations = (local_observations,active_queues)


        info = {}
        info['tasks_arrived'] = np.where(old_bitarrive==0,0,1)
        info['tasks_dropped'] = -np.ceil(rewards)
        info['rewards'] = rewards
        info['actions'] = actions
        info['bitarrive'] = old_bitarrive
        info['offloaded_tasks'] = np.array(np.logical_and(old_bitarrive,actions),dtype=np.int32)
        info['offloaded_rewards'] = offloaded_rewards
        info['offloaded_drop'] = -np.ceil(offloaded_rewards)


        return observations, rewards, done, info
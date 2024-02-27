

from environment.environment import Environment
from drl.agent import Agent
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch    

#DONT CHANGE
NUMBER_OF_CLOUDS = 1
np.random.seed(0)
def remove_id_from_list(lst, server_id):
    return lst[:server_id] + lst[server_id+1:]


if __name__ =='__main__':

    episodes  =1000
    number_of_servers =3
    servers_private_queues_computational_capacities = 2.5 *np.ones(number_of_servers)
    servers_public_queues_computational_capacities = 10 *np.ones(number_of_servers)
    transmission_capacities = 14 * np.ones([number_of_servers, (number_of_servers-1)+NUMBER_OF_CLOUDS])   # Mbps * duration
    cloud_computational_capacity = 30
    
    
    episode_time =100
    timeout_delay =10 
    max_bit_arrive = 5
    min_bit_arrive =2
    task_arrive_probability=0.4
    delta_duration=0.1
    task_drop_penalty_multiplier=4
    task_computational_density = 0.297
    environment = Environment(
                 servers_private_queues_computational_capacities=servers_private_queues_computational_capacities,
                 servers_public_queues_computational_capacities=servers_public_queues_computational_capacities,
                 transmission_capacities=transmission_capacities,
                 cloud_computational_capacity=cloud_computational_capacity,
                 episode_time=episode_time,
                 timeout_delay=timeout_delay,
                 max_bit_arrive=max_bit_arrive ,
                 min_bit_arrive=min_bit_arrive ,
                 task_arrive_probability=task_arrive_probability,
                 delta_duration=delta_duration,
                 task_drop_penalty_multiplier=task_drop_penalty_multiplier,
                 task_computational_density=task_computational_density,
                 number_of_clouds=NUMBER_OF_CLOUDS)
    
    
    state_dimensions,lstm_shape,number_of_actions = environment.get_agent_variables()
           
    hidden_layers =[100]
    lstm_layers = 20
    epsilon_decrement =1e-4
    batch_size =64
    learning_rate =1e-3
    memory_size = int(1e4)
    lstm_time_step = 10
    replace_target_iter = 500
    optimizer=optim.Adam
    loss_function=nn.MSELoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    agents = [Agent(
                    id =i,
                    state_dimensions=state_dimensions,
                    lstm_shape=lstm_shape,
                    number_of_actions=number_of_actions,
                    hidden_layers = hidden_layers,
                    lstm_layers = lstm_layers,
                    epsilon_decrement=epsilon_decrement,
                    batch_size = batch_size,
                    learning_rate=learning_rate,
                    memory_size=memory_size,
                    lstm_time_step=lstm_time_step,
                    replace_target_iter=replace_target_iter,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    device=device) 
        for i in range(number_of_servers)]
    
    scores = []
    for episode in range(episodes):
        done = False
        local_observations,active_queues = environment.reset()
        
        while not done:
            actions = np.zeros(number_of_servers,dtype=np.int8)
            for i in range(number_of_servers):
                lstm_input = remove_id_from_list(active_queues,i)
                actions[i] = agents[i].choose_action(local_observations[i],lstm_input)
            (local_observations_,active_queues_), rewards, done, info = environment.step(actions)
            score = np.mean(rewards)
            scores.append(score)
            new_lstm_input = remove_id_from_list(active_queues_,i)
            for i in range(number_of_servers):
                agents[i].store_transitions(state = local_observations[i],
                                            lstm_state=lstm_input,
                                            action = actions[i],
                                            reward= rewards[i],
                                            new_state=local_observations_[i],
                                            new_lstm_state=new_lstm_input,
                                            done=done)
                agents[i].learn()
                
            local_observations,active_queues  = local_observations_,active_queues_
            
        avg_score = np.mean(scores[-100:])
        print('Episode: {}\tScore: {}\t Average Score: {}\tEpsilon {}'.format(episode,score,avg_score,agents[0].epsilon))

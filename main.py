

from environment.environment import Environment
from drl.agent import Agent
import numpy as np

import torch.nn as nn
import torch    

import json
import argparse
import matplotlib.pyplot as plt 
import os 
#DONT CHANGE
NUMBER_OF_CLOUDS = 1
def remove_id_from_list(lst, server_id):
    return lst[:server_id] + lst[server_id+1:]


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
    parser.add_argument('--hyperparameters_file', type=str, default='metadata/hyperparameters.json', help='Hyperparameters File')

    args = parser.parse_args()
    hyperparameters_file = args.hyperparameters_file
        
    if not os.path.isfile(hyperparameters_file):
        os.system('python metadata/hyperparameter_generator.py')
    with open(hyperparameters_file, 'r') as file:
        hyperparameters = json.load(file)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print(device)
    for key in hyperparameters:
        print(key," : ",hyperparameters[key])
        
    episodes  =hyperparameters['episodes']
    number_of_servers = hyperparameters['number_of_servers']
    environment = Environment(
                servers_private_queues_computational_capacities = np.array(hyperparameters['servers_private_queues_computational_capacities']),
                servers_public_queues_computational_capacities = np.array(hyperparameters['servers_public_queues_computational_capacities']),
                transmission_capacities =  np.array(hyperparameters['transmission_capacities']),
                cloud_computational_capacity = hyperparameters['cloud_computational_capacity'],
                episode_time =hyperparameters['episode_time'],
                timeout_delay =hyperparameters ['timeout_delay'],
                max_bit_arrive = hyperparameters['max_bit_arrive'],
                min_bit_arrive =hyperparameters['min_bit_arrive'],
                task_arrive_probability=hyperparameters['task_arrive_probability'],
                delta_duration=hyperparameters['delta_duration'],
                task_drop_penalty_multiplier=hyperparameters['task_drop_penalty_multiplier'],
                task_computational_density = hyperparameters['task_computational_density'],
                 number_of_clouds=NUMBER_OF_CLOUDS)
    
    
    state_dimensions,lstm_shape,number_of_actions = environment.get_agent_variables()

    
    agents = [Agent(id =i,
                    state_dimensions=state_dimensions,
                    lstm_shape=lstm_shape,
                    number_of_actions=number_of_actions,
                    hidden_layers =hyperparameters['hidden_layers'],
                    lstm_layers = hyperparameters['lstm_layers'],
                    epsilon_decrement =hyperparameters['epsilon_decrement'],
                    batch_size =hyperparameters['batch_size'],
                    learning_rate =hyperparameters['learning_rate'],
                    memory_size = hyperparameters['memory_size'],
                    lstm_time_step = hyperparameters['lstm_time_step'],
                    replace_target_iter = hyperparameters['replace_target_iter'],
                    loss_function = getattr(nn, hyperparameters['loss_function']),
                    optimizer = getattr(torch.optim, hyperparameters['optimizer']),
                    device=device) 
        for i in range(number_of_servers)]
    
    scores_history = []
    drop_ratio_history =[]
    epsilon_history =[]
    for episode in range(episodes):
        done = False
        local_observations,active_queues = environment.reset()
        score =0
        tasks_arrived = 0
        tasks_dropped = 0
        
        while not done:
            actions = np.zeros(number_of_servers,dtype=np.int8)
            for i in range(number_of_servers):
                lstm_input = remove_id_from_list(active_queues,i)
                actions[i] = agents[i].choose_action(local_observations[i],lstm_input)
            (local_observations_,active_queues_), rewards, done, info = environment.step(actions)
            tasks_arrived += info['tasks_arrived']
            tasks_dropped +=info['tasks_dropped']
            score += np.mean(rewards)
            for i in range(number_of_servers):
                new_lstm_input = remove_id_from_list(active_queues_,i)
                agents[i].store_transitions(state = local_observations[i],
                                            lstm_state=lstm_input,
                                            action = actions[i],
                                            reward= rewards[i],
                                            new_state=local_observations_[i],
                                            new_lstm_state=new_lstm_input,
                                            done=done)
                agents[i].learn()
                
            local_observations,active_queues  = local_observations_,active_queues_
        
        scores_history.append(score)
        drop_ratio = tasks_dropped/tasks_arrived
        epsilon_history.append(agents[0].epsilon)
        drop_ratio_history.append(drop_ratio)
        avg_score = np.mean(scores_history[-100:])

        print('Episode: {}\tScore: {:.3f}\t Average Score: {:.3f}\tDrop Ratio: {:.3f}\tEpsilon: {:.3f}'.format(episode,score,avg_score,drop_ratio,agents[0].epsilon))
    plt.plot(epsilon_history)
    plt.plot(drop_ratio_history)
    plt.plot(scores_history)
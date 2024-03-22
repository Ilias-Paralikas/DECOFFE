

from environment.environment import Environment
from drl.agent import Agent
from bookkeeping.bookkeeper import Bookkeeper
import numpy as np

import torch    

import json
import argparse
import os 
#DONT CHANGE
NUMBER_OF_CLOUDS = 1
def remove_id_from_list(lst, server_id):
    return lst[:server_id] + lst[server_id+1:]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(device)
    parser = argparse.ArgumentParser(description="Process some integers.")
 
    parser.add_argument('--resume_run', type=str, nargs='?', default=None, help='an optional string')
    parser.add_argument('--episodes', type=int, default=1000, help='Integer')
    parser.add_argument('--average_window', type=int,   default=500, help='anerage ploting window')
    parser.add_argument('--log_folder' ,type=str,default='bookkeeping/log_folder',help='where the runs will be stored')
    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='the file that the hyperparameters will be saved, for verison control')
    parser.add_argument('--static',type=int, default=0, help='if the environment is static or not')
    args = parser.parse_args()
    resume_run = args.resume_run
    if resume_run:
        try:
            bookkeeper  =Bookkeeper(resume_run=resume_run,
                                    average_window=args.average_window,
                                    log_folder=args.log_folder)
        except:
            print('The run you are trying to resume does not exist')
            return
        hyperparameters_file,checkpoint_folder = bookkeeper.get_folder_names()
        with open(hyperparameters_file, 'r') as file:
            hyperparameters = json.load(file)
        hyperparameters['epsilon'] = bookkeeper.get_epsilon()
        hyperparameters['episodes'] = args.episodes
    else:
        hyperparameters_file = args.hyperparameters_file
        if not os.path.isfile(hyperparameters_file):
            os.system('python hyperparameters/hyperparameter_generator.py')
        with open(hyperparameters_file, 'r') as file:
            hyperparameters = json.load(file)
        bookkeeper  =Bookkeeper(resume_run=resume_run,
                                hyperparameters=hyperparameters,
                                average_window=args.average_window,
                                log_folder=args.log_folder)
        hyperparameters_file,checkpoint_folder = bookkeeper.get_folder_names()

    
    print(hyperparameters)
    
    
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
                    loss_function = getattr(torch.nn, hyperparameters['loss_function']),
                    optimizer = getattr(torch.optim, hyperparameters['optimizer']),
                    device=device,
                    checkpoint_folder =checkpoint_folder+'/agent_'+str(i)+'.pt' ,
                    gamma = hyperparameters['gamma'],
                    epsilon_end = hyperparameters['epsilon_end'],
                    local_action_probability = hyperparameters['local_action_probability'],
                    save_model_frequency = hyperparameters['save_model_frequency'],
                    epsilon=hyperparameters['epsilon'])
            for i in range(number_of_servers)]


    for episode in range(episodes):
        if args.static:
            if episode % args.static == 0:
                np.random.seed(0)
        done = False
        local_observations,active_queues = environment.reset()
        

        while not done:
            actions = np.zeros(number_of_servers,dtype=np.int8)
            for i in range(number_of_servers):
                lstm_input = remove_id_from_list(active_queues,i)
                actions[i] = agents[i].choose_action(local_observations[i],lstm_input)
            (local_observations_,active_queues_), rewards, done, info = environment.step(actions)
            bookkeeper.store_timestep(info)
            if not hyperparameters['validate']:
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
        bookkeeper.reset_episode(episode,agents[0].epsilon)

    bookkeeper.plot_metrics()
    
if __name__ =='__main__':
    main()


from environment.environment import Environment
from decision_makers.agent import Agent
from bookkeeping.bookkeeper import Bookkeeper
import numpy as np
from decision_makers.round_robbin import RoundRobin 
from decision_makers.random import Random
from decision_makers.uloof import ULOOF
import torch    

import json
import argparse
import os 
#DONT CHANGE
NUMBER_OF_CLOUDS = 1
def remove_id_from_list(lst, server_id):
    return lst[:server_id] + lst[server_id+1:]


def main():
    decision_makers = {
        'drl': Agent,
        'RoundRobin': RoundRobin,
        'Random':Random,
        'ULOOF':ULOOF
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(device)
    parser = argparse.ArgumentParser(description="Process some integers.")
 
    parser.add_argument('--resume_run', default=None,type=str, nargs='?', help='This argument is used to specify the run to resume. If this argument is provided, the script will attempt to resume a previous run with the given name. If the run does not exist, the script will print an error message and exit.')
    parser.add_argument('--episodes', default=10,type=int,  help='This argument specifies the number of episodes to run in the simulation. The default value is 10.')
    parser.add_argument('--average_window',default=500, type=int,   help=': This argument specifies the window size for averaging the results for plotting. The default value is 500.')
    parser.add_argument('--log_folder' ,type=str,default='bookkeeping/log_folder',help=' This argument specifies the directory where the logs will be stored. The default directory is "bookkeeping/log_folder"')
    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='This argument specifies the file where the hyperparameters are stored. The default file is "hyperparameters/hyperparameters.json"')
    parser.add_argument('--static',type=int, default=0, help='This argument specifies whether the environment is static or not. If this argument is set to a non-zero integer, the environment will be reset to its initial state every --static episodes.')
    parser.add_argument('--train_in_exploit_state', action="store_true", help='If this flag is set, the model will be trained in the exploit state.')
    parser.add_argument('--single_agent', default=None,type= str,help="This argument is used to specify a single agent's weights for all the agents. If this argument is provided, all agents will use the weights of the specified agent.")

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
    
    if args.resume_run  and args.single_agent:
        checkpoint_folders =  [checkpoint_folder+'/' + args.single_agent+'.pt' for i in range(number_of_servers)]
    else:
        checkpoint_folders = [checkpoint_folder+'/agent_'+str(i)+'.pt' for i in range(number_of_servers)]
        
    agents = [decision_makers[hyperparameters['descision_maker_choice']](id =i,
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
                    checkpoint_folder =checkpoint_folders[i] ,
                    gamma = hyperparameters['gamma'],
                    epsilon_end = hyperparameters['epsilon_end'],
                    local_action_probability = hyperparameters['local_action_probability'],
                    save_model_frequency = hyperparameters['save_model_frequency'],
                    epsilon=hyperparameters['epsilon'],
                    train_in_exploit_state = args.train_in_exploit_state,
                    hyperparameters=hyperparameters)
                    
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
            
        bookkeeper.reset_episode(episode,agents[0].get_epsilon())

    bookkeeper.plot_metrics()
    
if __name__ =='__main__':
    main()
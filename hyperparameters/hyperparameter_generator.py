import argparse
import json
import os 
NUMBER_OF_CLOUDS =1

def comma_seperated_string_to_list(comma_seperated_String):
        return [int(x) for x in comma_seperated_String.split(',')]

if __name__=="__main__":
        parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
        
        

        parser.add_argument('--servers_private_queues_computational_capacities', type=float, default=2.5, help='Float')
        parser.add_argument('--servers_public_queues_computational_capacities', type=float, default=5, help='Float')
        parser.add_argument('--horizontal_transmission_capacity', type=float, default=10, help='Float')
        parser.add_argument('--vertical_transmission_capacity', type=float, default=20, help='Float')
        parser.add_argument('--validate', action="store_true", help='if set, we will only validate the models, without training')

        
        parser.add_argument('--episodes', type=int, default=5, help='Integer')
        parser.add_argument('--number_of_servers', type=int, default=3, help='Integer')
        parser.add_argument('--cloud_computational_capacity', type=float, default=30, help='Float')

        # More hyperparameters
        parser.add_argument('--episode_time', type=int, default=100, help='Integer')
        parser.add_argument('--timeout_delay', type=int, default=10, help='Integer')
        parser.add_argument('--max_bit_arrive', type=float, default=5.0, help='Float')
        parser.add_argument('--min_bit_arrive', type=float, default=2.0, help='Float')
        parser.add_argument('--task_arrive_probability', type=float, default=0.4, help='Float between 0 and 1')
        parser.add_argument('--delta_duration', type=float, default=0.1, help='Float')
        parser.add_argument('--task_drop_penalty_multiplier', type=float, default=4, help='Float')
        parser.add_argument('--task_computational_density', type=float, default=0.297, help='Float')

        # Neural network hyperparameters
        parser.add_argument('--hidden_layers', type=str, default='20', help='comma-separated integers')
        parser.add_argument('--lstm_layers', type=int, default=20, help='Integer')
        parser.add_argument('--epsilon_decrement', type=float, default=1e-6, help='Float')
        parser.add_argument('--batch_size', type=int, default=64, help='Integer')
        parser.add_argument('--learning_rate', type=float, default=1e-7, help='Float')
        parser.add_argument('--memory_size', type=int, default=int(1e4), help='Integer')
        parser.add_argument('--lstm_time_step', type=int, default=10, help='Integer')
        parser.add_argument('--replace_target_iter', type=int, default=2000, help='Integer')
        parser.add_argument('--optimizer', type=str, default='Adam', help='selected from https://pytorch.org/docs/stable/optim.html#algorithms, provided as a string')
        parser.add_argument('--loss_function', type=str, default='MSELoss', help='selected from https://pytorch.org/docs/stable/nn.html#loss-functions, provided as a string, without the nn')
        parser.add_argument('--gamma', type=float, default=0.8, help='Float')
        parser.add_argument('--epsilon_end', type=float, default=0.01, help='float, between 0 and 1')
        parser.add_argument('--local_action_probability', type=float, default=0.5, help='Float when picking random action, the probability of chosing local actions')
        parser.add_argument('--save_model_frequency', type=int, default=100, help='Integer, How ofter should the models be saved')
        parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='the file that the hyperparameters will be saved, for verison control')
        args = parser.parse_args()
        if args.validate:
                epsilon = 0.0
        else:
                epsilon = 1.0
                

        
        hidden_layers = comma_seperated_string_to_list(args.hidden_layers)
        hyperparameters = {
        'episodes': args.episodes,
        'number_of_servers': args.number_of_servers,
        'cloud_computational_capacity': args.cloud_computational_capacity,
        'episode_time': args.episode_time,
        'timeout_delay': args.timeout_delay,
        'max_bit_arrive': args.max_bit_arrive,
        'min_bit_arrive': args.min_bit_arrive,
        'task_arrive_probability': args.task_arrive_probability,
        'delta_duration': args.delta_duration,
        'task_drop_penalty_multiplier': args.task_drop_penalty_multiplier,
        'task_computational_density': args.task_computational_density,
        'hidden_layers': hidden_layers,
        'lstm_layers': args.lstm_layers,
        'epsilon_decrement': args.epsilon_decrement,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'memory_size': args.memory_size,
        'lstm_time_step': args.lstm_time_step,
        'replace_target_iter': args.replace_target_iter,
        'optimizer': args.optimizer,
        'loss_function': args.loss_function,
        'validate':args.validate,
        'epsilon':epsilon,
        'gamma':args.gamma,
        'epsilon_end':args.epsilon_end,
        'local_action_probability':args.local_action_probability,
        'save_model_frequency' :args.save_model_frequency
        }


        hyperparameters['servers_private_queues_computational_capacities'] = [args.servers_private_queues_computational_capacities for _ in range(hyperparameters['number_of_servers'])]
        hyperparameters['servers_public_queues_computational_capacities'] = [args.servers_public_queues_computational_capacities for _ in range(hyperparameters['number_of_servers'])]
        hyperparameters['transmission_capacities'] = [[args.horizontal_transmission_capacity for _ in range(hyperparameters['number_of_servers']-1 + NUMBER_OF_CLOUDS)] for _ in range(hyperparameters['number_of_servers'])]
        for row in hyperparameters['transmission_capacities']:
                row[-1] = args.vertical_transmission_capacity

        json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)
        
        with open(args.hyperparameters_file, "w") as outfile:
                outfile.write(json_object)
        

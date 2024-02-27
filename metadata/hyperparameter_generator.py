import argparse
import json

NUMBER_OF_CLOUDS =1

def comma_seperated_string_to_list(comma_seperated_String):
        return [int(x) for x in comma_seperated_String.split(',')]

if __name__=="__main__":
        parser = argparse.ArgumentParser(description='Script Configuration via Command Line')

        parser.add_argument('--servers_private_queues_computational_capacities', type=float, default=2.5, help='Servers private queues computational capacities')
        parser.add_argument('--servers_public_queues_computational_capacities', type=float, default=5, help='Servers public queues computational capacities')
        parser.add_argument('--horizontal_transmission_capacity', type=float, default=10, help='Horizontal transmission capacity')
        parser.add_argument('--vertical_transmission_capacity', type=float, default=20, help='Vertical transmission capacity')
        
        
        parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
        parser.add_argument('--number_of_servers', type=int, default=3, help='Number of servers')
        parser.add_argument('--cloud_computational_capacity', type=float, default=30, help='Cloud computational capacity')

        # More hyperparameters
        parser.add_argument('--episode_time', type=int, default=100, help='Episode time')
        parser.add_argument('--timeout_delay', type=int, default=10, help='Timeout delay')
        parser.add_argument('--max_bit_arrive', type=float, default=5.0, help='Maximum bit arrive')
        parser.add_argument('--min_bit_arrive', type=float, default=2.0, help='Minimum bit arrive')
        parser.add_argument('--task_arrive_probability', type=float, default=0.4, help='Task arrive probability')
        parser.add_argument('--delta_duration', type=float, default=0.1, help='Delta duration')
        parser.add_argument('--task_drop_penalty_multiplier', type=int, default=4, help='Task drop penalty multiplier')
        parser.add_argument('--task_computational_density', type=float, default=0.297, help='Task computational density')

        # Neural network hyperparameters
        parser.add_argument('--hidden_layers', type=str, default='100', help='Hidden layers sizes, comma-separated')
        parser.add_argument('--lstm_layers', type=int, default=20, help='LSTM layers')
        parser.add_argument('--epsilon_decrement', type=float, default=1e-4, help='Epsilon decrement')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--memory_size', type=int, default=int(1e4), help='Memory size')
        parser.add_argument('--lstm_time_step', type=int, default=10, help='LSTM time step')
        parser.add_argument('--replace_target_iter', type=int, default=500, help='Replace target iteration')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
        parser.add_argument('--loss_function', type=str, default='MSELoss', help='Loss function')
        
        parser.add_argument('--hyperparameters_folder', type=str, default='metadata/hyperparameters.json', help='Hyperparameters Folder')


        
        args = parser.parse_args()

        
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
        'loss_function': args.loss_function
        }


        hyperparameters['servers_private_queues_computational_capacities'] = [args.servers_private_queues_computational_capacities for _ in range(hyperparameters['number_of_servers'])]
        hyperparameters['servers_public_queues_computational_capacities'] = [args.servers_public_queues_computational_capacities for _ in range(hyperparameters['number_of_servers'])]
        hyperparameters['transmission_capacities'] = [[args.horizontal_transmission_capacity for _ in range(hyperparameters['number_of_servers']-1 + NUMBER_OF_CLOUDS)] for _ in range(hyperparameters['number_of_servers'])]
        for row in hyperparameters['transmission_capacities']:
                row[-1] = args.vertical_transmission_capacity

        json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)
        
        with open(args.hyperparameters_folder, "w") as outfile:
                outfile.write(json_object)
        

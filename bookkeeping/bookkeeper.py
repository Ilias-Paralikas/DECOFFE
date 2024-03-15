import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
class Bookkeeper:
    def __init__(self,
                 resume_run,
                 average_window,
                 log_folder,
                 hyperparameters=None):
        self.average_window=average_window
        os.makedirs(log_folder,exist_ok=True)
        if resume_run:
            self.run_folder = log_folder+'/'+resume_run
            self.hyperparameters_file = self.run_folder+'/hyperparameters.json'

        else:
            index_filepath  =log_folder+'/index.txt'
            if not os.path.exists(index_filepath):
                with open(index_filepath, 'w') as file:
                    file.write('0')
                    run_index = 0
            else:
                with open(index_filepath, 'r') as file:
                    run_index = int(file.read().strip())
                run_index += 1
                with open(index_filepath, 'w') as file:
                    file.write(str(run_index))
            self.run_folder = log_folder+'/run_'+str(run_index)
            os.makedirs(self.run_folder,exist_ok=True)
    
            self.hyperparameters_file = self.run_folder+'/hyperparameters.json'
            json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)
            with open(self.hyperparameters_file, "w") as outfile:
                    outfile.write(json_object)
        
        self.checkpoint_folder = self.run_folder+'/checkpoints'
        self.metrics_folder = self.run_folder+'/metrics.pkl'
        
                
        if resume_run:
            with open(self.metrics_folder, 'rb') as f:
                self.metrics = pickle.load(f)
        else:
            self.metrics ={}
            self.metrics['task_drop_ratio_history'] =[]
            self.metrics['rewards_history'] =[]
            self.metrics['epsilon_history'] =[1.0]

    
        self.tasks_arrived = []
        self.tasks_dropped =[]        
        self.rewards = []
        
        os.makedirs(self.checkpoint_folder,exist_ok=True)

    def reset_episode(self,episode,epsilon):
        episode_tasks_arrived = np.vstack(self.tasks_arrived)
        episode_tasks_arrived = np.sum(episode_tasks_arrived,axis=0)
        episode_tasks_drop = np.vstack(self.tasks_dropped)
        episode_tasks_drop = np.sum(episode_tasks_drop,axis=0)
        episode_task_drop_ratio = episode_tasks_drop/episode_tasks_arrived
        self.metrics['task_drop_ratio_history'].append(episode_task_drop_ratio)
        
        episode_rewards = np.vstack(self.rewards)
        episode_rewards=  np.sum(episode_rewards,axis=0)
        self.metrics['rewards_history'].append(episode_rewards)
          
        self.metrics['epsilon_history'].append(epsilon)
        
                
        with open(self.metrics_folder, 'wb') as f:
            pickle.dump(self.metrics, f)
            
        self.tasks_arrived = []
        self.tasks_dropped =[]
        self.rewards = []
        score, average_score,drop_ratio,epsilon = np.mean(self.metrics['rewards_history'][-1]), np.mean(self.metrics['rewards_history'][-self.average_window:]),np.mean(self.metrics['task_drop_ratio_history'][-1]),self.metrics['epsilon_history'][-1]
        print('Episode: {}\tScore: {:.3f}\t Average Score: {:.3f}\tDrop Ratio: {:.3f}\tEpsilon: {:.3f}'.format(episode,score,average_score,drop_ratio ,epsilon))
    def store_timestep(self,info):
        self.tasks_arrived.append(info['tasks_arrived'])
        self.tasks_dropped.append(info['tasks_dropped'])
        self.rewards.append(info['rewards'])
            
        



    def get_folder_names(self):
        return self.hyperparameters_file,self.checkpoint_folder
    def get_epsilon(self):
        return self.metrics['epsilon_history'][-1]
    
    
    def plot_and_save(self, key):
        if key not in self.metrics:
            print(f"No data found for key '{key}'")
            return
        list_of_arrays = self.metrics[key]
        stacked_arrays = np.vstack(list_of_arrays)

        transposed_arrays = stacked_arrays.T
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(transposed_arrays):
            plt.plot(column, label=f'agent {i+1} {key} ', linestyle='--')
        mean_values = np.mean(transposed_arrays, axis=0)
        plt.plot(mean_values, label='Mean', color='red', linewidth=6)
        plt.legend()
        plt.title(f'Plot of {key} and Their Mean')

        plt.savefig(f'{self.run_folder}/{key}.png')

    def moving_average(self, a):
        return  [np.mean(a[max(0,i-self.average_window):i]) for i in range(1,len(a))]

        
    def plot_and_save_moving_avg(self, key):
        if key not in self.metrics:
            print(f"No data found for key '{key}'")
            return

        list_of_arrays = self.metrics[key]

        stacked_arrays = np.vstack(list_of_arrays)

        transposed_arrays = stacked_arrays.T

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot the moving average of each column
        means = []  # List to store the means of the moving averages
        for i, column in enumerate(transposed_arrays):
            moving_avg = self.moving_average(column)  # Change n to your desired window size
            plt.plot(moving_avg, label=f'agent {i+1}', linestyle='--')
            means.append(moving_avg)
        means = np.mean(means, axis=0)
        # Plot the mean of the moving averages
        plt.plot(means, label='Mean', color='red',linewidth=6)

        # Add a legend and title
        plt.legend()
        plt.title(f'Plot of Moving Average of {key}')

        # Save the plot as a PNG file
        plt.savefig(f'{self.run_folder}/{key}_moving_average.png')
        plt.close()
        
    def plot_metrics(self):
        for key in self.metrics.keys():
            self.plot_and_save(key)
            self.plot_and_save_moving_avg(key)
        
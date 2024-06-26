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
            self.metrics['offloaded_drop_ratio_history'] =[]
            self.metrics['rewards_history'] =[]
            self.metrics['offloaded_rewards_history'] = []
            self.metrics['epsilon_history'] =[1.0]
            self.metrics['lr_history'] =[hyperparameters['learning_rate']]
            self.metrics['champion_score']  = float('-inf')


    
        self.tasks_arrived = []
        self.tasks_offloaded = []
        self.tasks_dropped =[]    
        self.tasks_offloaded_dropped = []    
        self.rewards = []
        self.offloaded_rewards = []
        self.actions = []
        
        os.makedirs(self.checkpoint_folder,exist_ok=True)
        
        self.not_plotable_metrics  = ['champion_score']
        

    def reset_episode(self,epsilon,learning_rate):
        episode_tasks_arrived = np.vstack(self.tasks_arrived)
        episode_tasks_arrived = np.sum(episode_tasks_arrived,axis=0)
        episode_tasks_drop = np.vstack(self.tasks_dropped)
        episode_tasks_drop = np.sum(episode_tasks_drop,axis=0)
        episode_task_drop_ratio = episode_tasks_drop/episode_tasks_arrived
        self.metrics['task_drop_ratio_history'].append(episode_task_drop_ratio)
        
        
        episode_tasks_offloaded = np.vstack(self.tasks_offloaded)
        episode_tasks_offloaded = np.sum(episode_tasks_offloaded,axis=0)
        episode_tasks_offloaded_drop = np.vstack(self.tasks_offloaded_dropped)
        episode_tasks_offloaded_drop = np.sum(episode_tasks_offloaded_drop,axis=0)
        episode_offloaded_drop_ratio = episode_tasks_offloaded_drop/episode_tasks_offloaded
        self.metrics['offloaded_drop_ratio_history'].append(episode_offloaded_drop_ratio)
        
        episode_rewards = np.vstack(self.rewards)
        episode_rewards=  np.sum(episode_rewards,axis=0)
        self.metrics['rewards_history'].append(episode_rewards)
        
        episode_offloaded_rewards = np.vstack(self.offloaded_rewards)
        episode_offloaded_rewards=  np.sum(episode_offloaded_rewards,axis=0)
        self.metrics['offloaded_rewards_history'].append(episode_offloaded_rewards)
        
          
        self.metrics['epsilon_history'].append(epsilon)
        self.metrics['lr_history'].append(learning_rate)
        
      
        with open(self.metrics_folder, 'wb') as f:
            pickle.dump(self.metrics, f)
            
        self.tasks_arrived = []
        self.tasks_offloaded = []
        self.tasks_dropped =[]
        self.tasks_offloaded_dropped = []
        self.rewards = []
        self.offloaded_rewards = []
        self.last_actions = self.actions
        self.actions = []
        score, average_score,drop_ratio,epsilon = np.mean(self.metrics['rewards_history'][-1]), np.mean(self.metrics['rewards_history'][-self.average_window:]),np.mean(self.metrics['task_drop_ratio_history'][-1]),self.metrics['epsilon_history'][-1]
        print('Episode: {}\tScore: {:.3f}\t Average Score: {:.3f}\tDrop Ratio: {:.3f}\tEpsilon: {:.3f}'.format(len(self.metrics['rewards_history']),score,average_score,drop_ratio ,epsilon))
    
    def get_agent_average_scores(self):
        return np.array(self.metrics['rewards_history'][-self.average_window:]).mean(axis=0)
    def start_championship(self,championship_epsilon_start,championship_episode_start):
        self.championship_epsilon_start = championship_epsilon_start
        self.championship_episode_start = championship_episode_start
    def get_champion(self,
                     running_episode):
        
        total_episodes =len(self.metrics['rewards_history'])
        if self.metrics['epsilon_history'][-1] < self.championship_epsilon_start and running_episode >self.championship_episode_start:
            current_average_scores =  np.array(self.metrics['rewards_history'][-self.average_window:]).mean(axis=0)  
            best_score = current_average_scores.max()
            champion_status = np.where((current_average_scores == best_score) &(best_score > self.metrics['champion_score']), True, False)
            if best_score > self.metrics['champion_score']:
                self.metrics['champion_score']  = best_score
            return champion_status, total_episodes
        else:
            return [False]*len(self.metrics['rewards_history'][-1]) ,total_episodes

    def store_timestep(self,info):
        self.tasks_arrived.append(info['tasks_arrived'])
        self.tasks_offloaded.append(info['offloaded_tasks'])
        self.tasks_dropped.append(info['tasks_dropped'])
        self.tasks_offloaded_dropped.append(info['offloaded_drop'])
        self.rewards.append(info['rewards'])
        self.offloaded_rewards.append(info['offloaded_rewards'])
        valid_actions  =  [info['actions'][i] if info['bitarrive'][i] !=0 else -1 for i in range(len(info['bitarrive']))]
        self.actions.append(valid_actions)
            
        



    def get_folder_names(self):
        return self.hyperparameters_file,self.checkpoint_folder
    def get_epsilon(self):
        return self.metrics['epsilon_history'][-1]
    def get_learning_rate(self):   
        return self.metrics['lr_history'][-1]
    def plot_actions(self):
        actions = np.concatenate(self.last_actions)
        actions = [num for num in actions if num != -1]
        counts = np.bincount(actions)
        plt.bar(range(len(counts)), counts)
        plt.title(f'actions chosen by agents')

        plt.savefig(f'{self.run_folder}/actions.png')
    
    
    def plot_and_save(self, key):
        if key not in self.metrics:
            print(f"No data found for key '{key}'")
            return
        if key in self.not_plotable_metrics:
            return 
        list_of_arrays = self.metrics[key]
        stacked_arrays = np.vstack(list_of_arrays)

        transposed_arrays = stacked_arrays.T
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(transposed_arrays):
            plt.plot(column, label=f'agent {i} {key} ', linestyle='--')
        mean_values = np.mean(transposed_arrays, axis=0)
        plt.plot(mean_values, label='Mean', color='red', linewidth=6)
        plt.legend()
        plt.title(f'Plot of {key} and Their Mean')

        plt.savefig(f'{self.run_folder}/{key}.png')
        plt.close()

    def moving_average(self, a):
        return  [np.mean(a[max(0,i-self.average_window):i]) for i in range(1,len(a))]

        
    def plot_and_save_moving_avg(self, key):
        if key not in self.metrics:
            print(f"No data found for key '{key}'")
            return
        if key in self.not_plotable_metrics:
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
            plt.plot(moving_avg, label=f'agent {i}', linestyle='--')
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
        self.plot_actions()

        
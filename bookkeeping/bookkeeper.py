import numpy as np 
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import json
class Bookkeeper:
    def __init__(self,log_folder,hyperparameters,device='cpu'):
        self.reset_episode()
        self.total_score_history =[]
        self.score_history =[]
        self.average_score_history =[]
        
        self.total_drop_ratio_history =[]
        self.drop_ratio_history =[]
        self.epsilon_history =[]
        
        self.hyperparameters =hyperparameters
        self.log_folder = log_folder
        
        os.makedirs(log_folder, exist_ok=True)
        print(device)
        for key in hyperparameters:
            print(key," : ",hyperparameters[key])
    

        
        

        
    def store_episode(self,epsilon):
        self.epsilon_history.append(epsilon)
        self.total_score_history.append(self.total_step_score)
        self.score_history.append(self.step_score)
        
        total_episode_drop_ratio = self.total_step_tasks_dropped/self.total_step_tasks_arrived 
        self.total_drop_ratio_history.append(total_episode_drop_ratio)
        
        episode_drop_ratio =  self.step_tasks_dropped/self.step_tasks_arrived 
        self.drop_ratio_history.append(episode_drop_ratio)
        
        
        average_score = np.mean(self.total_score_history[-100:])
        self.average_score_history.append(average_score)
        
        episode_metrics = {}
        episode_metrics['score'] = self.total_step_score
        episode_metrics['avergae_score']  = average_score
        episode_metrics['drop_ratio'] = total_episode_drop_ratio

        self.reset_episode()
        
        return episode_metrics['score'],episode_metrics['avergae_score'],episode_metrics['drop_ratio'] 

    def reset_episode(self):
        self.total_step_score=0
        self.step_score = []
        
        self.total_step_tasks_arrived = 0
        self.step_tasks_arrived = []
        
        self.total_step_tasks_dropped = 0
        self.step_tasks_dropped = []

    
    def add_time_step(self,rewards,info):
        self.total_step_score += np.mean(rewards)
        if len(self.step_score)==0:
            self.step_score =rewards
        else:
            self.step_score += rewards
            
        self.total_step_tasks_arrived += info['total_tasks_arrived']
        if len(self.step_tasks_arrived)==0:
            self.step_tasks_arrived = info['tasks_arrived']
        else:
            self.step_tasks_arrived += info['tasks_arrived']
            
        self.total_step_tasks_dropped += info['total_tasks_dropped']
        if len(self.step_tasks_dropped)==0:
            self.step_tasks_dropped = info['tasks_dropped']
        else:  
            self.step_tasks_dropped += info['tasks_dropped']

        

        
    def store_run(self,index_filepath = 'run_index.txt'):
        metrics ={}
        metrics['total_score_history'] =self.total_score_history
        metrics['score_history']= self.score_history
        metrics['total_drop_ratio_history'] = self.total_drop_ratio_history
        metrics['drop_ratio_history'] = self.drop_ratio_history
        metrics['epsilon_history']  =self.epsilon_history
        
        run_details ={}
        run_details['hyperparameters'] =self.hyperparameters
        run_details['metrics'] =metrics
        
        index_filepath  = self.log_folder +'/' +index_filepath
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

        run_folder=  self.log_folder+'/run_'+str(run_index)
        os.mkdir(run_folder)
                
        filename = run_folder+'/hyperparameters.json'

            
        json_object = json.dumps(self.hyperparameters,indent=4) ### this saves the array in .json format)
        
        with open(filename, "w") as outfile:
                outfile.write(json_object)
        
                        
        with open(run_folder+'/details.pickle', 'wb') as handle:
            pickle.dump(run_details, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        

        linestyles = itertools.cycle(('-','--','-.',':'))
        
        plt.figure(figsize=(20,20))
        plt.plot(metrics['total_score_history'],label='Mean score',linestyle=next(linestyles))
        for agent_id in range(len(metrics['score_history'][0])):
            agent_score = [row[agent_id] for row in metrics['score_history']]
            plt.plot(agent_score,label='agent '+str(agent_id) + 'score',linestyle=next(linestyles))
        plt.legend()
        plt.savefig(run_folder+'/total_score_history')
        plt.close()
        

        linestyles = itertools.cycle(('-','--','-.',':'))
        plt.figure(figsize=(20,20))
        plt.plot(metrics['total_drop_ratio_history'],label='Total drop ratio',linestyle=next(linestyles))
        for agent_id in range(len(metrics['drop_ratio_history'][0])):
            agent_drop_rate = [row[agent_id] for row in metrics['drop_ratio_history']]
            plt.plot(agent_drop_rate,label='agent '+str(agent_id) + ' drop ratio',linestyle=next(linestyles))
        plt.legend()
        plt.savefig(run_folder+'/total_drop_ratio_history')
        plt.close()

        plt.figure(figsize=(20,20))
        plt.plot(self.average_score_history,label='average score')
        plt.legend()
        plt.savefig(run_folder+'/average_score')
        plt.close()
    

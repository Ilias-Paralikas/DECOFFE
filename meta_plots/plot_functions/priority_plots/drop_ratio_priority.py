import os
import json
import pickle
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import csv



folder ='meta_plots/logs/priority_results/priorit_logs/baselines'
runs_folder = os.path.join(folder ,'runs')

def group_metrics(log_folder, key, n):
    def transpose_2d_list(lst):
        return [list(i) for i in zip(*lst)]
    all_averages = {}  # Initialize all_averages as a dictionary
    subfolders = [f.path for f in os.scandir(log_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Read hyperparameters.json
        hyperparameters_file = os.path.join(subfolder, 'hyperparameters.json')
        with open(hyperparameters_file, 'r') as f:
            hyperparameters = json.load(f)
        server_priorities = hyperparameters['server_priorities']

        # Group server_priorities
        server_priorities_groups = [list(group) for key, group in groupby(server_priorities)]

        metrics_file = os.path.join(subfolder, 'metrics.pkl')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                run_metrics = pickle.load(f)
                if key in run_metrics:
                    # Get the last n values of the key
                    values = run_metrics[key][-n:]
                    values= transpose_2d_list(values)
                    averages = []
                    for group in server_priorities_groups:
                        group_values = values[:len(group)]
                        values = values[len(group):]  # Remove the used values
                        average = np.mean([item for sublist in group_values for item in sublist])
                        averages.append(average)

        all_averages[os.path.basename(subfolder)] = np.array(averages)  # Store averages for the current subfolder

    total_priorities = len(server_priorities_groups)
    return all_averages, total_priorities 
import matplotlib.pyplot as plt

def plot_dict(d,filename):
    plt.figure()  # Create a new figure
    for key, values in d.items():
        plt.plot([i+1 for i in range(len(values))], values, label=key)
    plt.xlabel('Priority Class')
    plt.ylabel('Drop Ratio')
    plt.legend()
    plt.savefig(filename)
    

    
def write_to_csv(sorted_items, filename, total_priorities,x_axis_name='Number of Servers'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([x_axis_name] + [f'Priority {i+1}' for i in range(total_priorities)])
        for item in sorted_items:
            writer.writerow([item[0]] + list(item[1]))

# Use the function
averages, total_priorities  = group_metrics(runs_folder, 'task_drop_ratio_history', 500)
plot_dict(averages,os.path.join(folder,'task_drop_ratio_history.png'))
write_to_csv(averages.items(),os.path.join(folder,'task_drop_ratio_history.csv'), total_priorities)



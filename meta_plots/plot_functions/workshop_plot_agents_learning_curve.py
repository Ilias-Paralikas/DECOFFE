import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

def replace_keys(old_dict, key_mapping):
    return {key_mapping.get(k, k): v for k, v in old_dict.items()}

def load_all_metrics(log_folder):
    metrics = {}
    run_folders = os.listdir(log_folder)

    for run_folder in run_folders:
        metrics_file = os.path.join(log_folder,run_folder, 'metrics.pkl')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                metrics[os.path.basename(run_folder)] = pickle.load(f)

    return metrics
def moving_average(a,average_window):
    return  [np.mean(a[max(0,i-average_window):i]) for i in range(1,len(a))]

def main():
    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')

    parser.add_argument('--plot_value', type=str, default ='rewards_history',help='name of the metric you want to plot. Note it must match the name in the metrics.pkl file')
    parser.add_argument('--folder', type=str, default='meta_plots/logs/workshop/workshop_logs/gamma', help='path to the folder containing the logs')
    parser.add_argument('--average_window', type=int, default=2000)
    
    args = parser.parse_args()  # Parse the command line arguments
    plot_value = args.plot_value  # Get the plot_value from the command line arguments
    folder = os.path.join(args.folder ,'runs') # Get the folder from the command line arguments
    average_window = args.average_window  # Set the size of the moving average window

    # key_mapping = {'run_1': 'α=10⁻³', 'run_2': 'α=10⁻⁴','run_3': 'α=10⁻⁵'}
    key_mapping = {'run_0': 'γ=0.8', 'run_1': 'γ=0.9','run_2': 'γ=0.99'}
    metrics =  load_all_metrics(folder)
    metrics = replace_keys(metrics, key_mapping)

    plt.rcParams['font.size'] = 17
    plt.figure(figsize=(8, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors
    linestyles = ['-', '--', '-.', ':']  # Define a list of linestyles

    for j, m in enumerate(metrics):
        values = metrics[m][plot_value]
        stacked_arrays = np.vstack(values)

        transposed_arrays = stacked_arrays.T
        for i, column in enumerate(transposed_arrays):
            moving_avg = moving_average(column,average_window)  # Change n to your desired window size
            plt.plot(moving_avg, label=f'{m} agent {i} ', color=colors[j],linestyle=linestyles[j],linewidth = 2)
    plt.legend(loc='lower right')
    
    
    plt.xlabel('Episodes',fontsize = 25)  # Replace with your actual x axis name
    plt.ylabel('Reward',fontsize = 25)
    plt.savefig(os.path.join(folder, args.plot_value+'.png'),dpi=500, bbox_inches='tight')

        
if __name__ == '__main__':
    main() 
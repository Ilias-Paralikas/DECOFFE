import json
import argparse

def add_hyperparameter(file, parameter, value):
    filename = file+'/hyperparameters.json'

    with open(filename, 'r') as f:
        data = json.load(f)
        
    data[parameter] = value

    with open(filename, 'w') as f:
        json.dump(data, f)
        
        
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', type=str, help='The path to the hyperparameters file')
    parser.add_argument('--parameter', type=str, help='The parameter to add')
    parser.add_argument('--value', type=str, help='The value to add')
    parser.parse_args()
    add_hyperparameter(parser.file, parser.parameter, parser.value)

            
if __name__ == "__main__":
    main()
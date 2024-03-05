# DECOFFE

Installation 
1. Clone the repository:
```
git clone https://github.com/Ilias-Paralikas/DECOFFE.git
```
2. Navigate to the project directory:
```
cd DECOFFE
```
3. Create a virtual environment:
```
python3 -m venv venv
```
4. Activate the virtual environment:
- On Windows:
```
  venv\Scripts\activate
```
- On Unix or MacOS:
```
  source venv/bin/activate
```
5. Install the requirements:
```
pip install -r requirements.txt
```

How to use
Define the hyperparameters
```
python DECOFFE/hyperparameters/hyperparameter_generator.py \
 --hyperparameters_folder DECOFFE/hyperparameters/hyperparameters.json \
```
Run the model
```
python DECOFFE/main.py \
--hypermarameters_folder DECOFFE/hyperparameters/hyperparameters.json 
```
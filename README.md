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
python -m venv venv
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

Run the model with the preset hyperparameters
```
python main.py 
```

If you want to modify the hyperparameters run
```
python hyperparameters/hyperparameter_generator.py --help
```

Note 
all paths are relative to ```main.py```
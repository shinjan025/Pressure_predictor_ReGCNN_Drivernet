# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:26:38 2024

Training script to train a GCNN to predict pressure distribution over a vehicle surface 
Config file is used to provide hyper-params and path to folder containing surface VTK files and pressure scalars
for ~8000 points 
best trained model is saved for future testing. 

@author: shinj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:40:41 2024

@author: shinj
"""

import torch
from model_trainer import run_training, setup_seed
from model import RegDGCNN
from torch.utils.data import DataLoader
from class_data_arrangements import make_data_set, 

config = {
    'exp_name': 'CdPrediction_DrivAerNet_r2_100epochs_5k',
    'path' : "C:/Users/shinj/Documents/PressureVTK/",
    'path_val' : "C:/Users/shinj/Documents/PressureVTK/Val_data",
    'cuda': True,
    'seed': 1,
    'num_points': 4,   #no of data points
    'num_points_val': 9,
    'lr': 0.001,
    'batch_size': 100,
    'epochs': 2,
    'dropout': 0.4,
    'emb_dims': 512,
    'k': 3,
    'optimizer': 'adam',
    'sample_size':1000,       # size of sample for each data pt
    'sample_size_val':1000,
    
    'V_norm':30,
    'dim':3
       
   
}


#specify device 
device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
setup_seed(config['seed'])

#initialize model 
model = RegDGCNN(config)

training_data = make_data_set(path=config["path"], sample_size=config["sample_size"], n_points=config["num_points"], V_norm=config["V_norm"])
#create training data based on path
training_data.Make_training_data()

#normalize training data based on physics based length scales
training_data.normalization_physics()

print(f">>>>>Number of data points : {training_data.input_points.shape[0]}")
print(f">>>>>Number of input features : {training_data.input_points.shape[1]}")

#convert point cloud to pytorch tensor
points = torch.from_numpy(training_data.input_points).to(device).float()

#reshape the point cloud into a 3D tensor
points = points.reshape(config['sample_size'],config['dim'],config['num_points']) 

#convert label to pytorch tenso
label = torch.from_numpy(training_data.output_points).to(device).float()

#create indexed dataset
train_dataloader = Data_4_training(points, label)

#create mini batching with shuffle
training_data = DataLoader(train_dataloader, batch_size=config["batch_size"], shuffle=True)

#training data 
model=run_training(model, training_data, config)




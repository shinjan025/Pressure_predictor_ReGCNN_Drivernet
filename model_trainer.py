# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:44:11 2024

@author: shinj
"""

import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
#from torchvision import transforms
from tqdm import tqdm
from model import RegDGCNN
#from DrivAerNetDataset import DrivAerNetDataset
import pandas as pd




# Set the device for train



def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def calc_val_loss(model, input, true_output):
    "compute validation loss"
    with torch.no_grad():
        pred_output = model(input)
         
    val_loss=F.mse_loss(pred_output.squeeze(), true_output)
    return val_loss.item()


def run_training(model: torch.nn.Module, train_dataloader: DataLoader, config: dict):
    """


    """
    train_losses= []
    training_start_time = time.time()  # Start timing for training

    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4) if config[
                                                                                          'optimizer'] == 'adam' else optim.SGD(
        model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    # Initialize the learning rate scheduler (ReduceLROnPlateau) to reduce the learning rate based on validation loss
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

    best_mse = float('inf')  # Initialize the best MSE as infinity
    
    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")


    # Training loop over the specified number of epochs
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Start timing for this epoch
        model.train()  # Set the model to training mode
        total_loss, total_r2 = 0, 0

        # Iterate over the training data
        for data, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data, targets = data.to(device), targets.to(device).squeeze()  # Move data to the gpu
           

            optimizer.zero_grad()
            outputs, features = model(data)
            #print("shapes of loss ",outputs.shape,targets.shape)

            loss = F.mse_loss(outputs.squeeze(), targets)
            
            r2 = r2_score(outputs.squeeze(), targets)  # Compute R2 score
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
            total_r2 += r2.item()
            print(features.shape)
        epoch_duration = time.time() - epoch_start_time
        # Calculate and print the average training loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        print("average Training loss : ", avg_loss)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s")

        avg_r2 = total_r2 / len(train_dataloader)
        print(f"Average Training R²: {avg_r2:.4f}")



    training_duration = time.time() - training_start_time
    print(f"Total training time: {training_duration:.2f}s")
    # Save the final model state to disk
    #model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    torch.save(model.state_dict(),'best_model.pt') #, model_path)
    print(f"Model saved ")
    # Save losses for plotting
    np.save('train_loss.npy',np.array(train_losses))
    return model



    
    
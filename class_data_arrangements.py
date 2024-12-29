# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:59:11 2024

@author: shinj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:12:34 2024

File to generate training data set for creating the subsampled training data for the DGCNN. 
@author: shinj
"""

import vtk 
import os
from random import sample
import pandas as pd
from os import listdir
from os.path import isfile, join
import vtk.util.numpy_support

import numpy as np
import torch


#mypath ="C:/Users/shinj/Documents/PressureVTK/"

class make_data_set():
    
 def __init__(self, path, sample_size, n_points,V_norm):
        self.input_points = None
        self.output_points = None
        self.path=path
        self.V_norm=V_norm
        self.sample_size=sample_size
        self.n_points=n_points
        self.L_norm=None
        

        



#create training inputs and output datasets
 def Make_training_data(self):     
    
   allfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]

   files = sample(allfiles,self.n_points)

 #create dataframe for overall training data

   df_subsampled = pd.DataFrame()
   # function to extract data from vtk with array names

   def get_vtk_array(data, array_name):
     array=data.GetPointData().GetArray(array_name)
     if array is None:
           raise ValueError("f'{Array_Name}'not found in this VTK file.")
     return vtk.util.numpy_support.vtk_to_numpy(array)
   
   for file in files:
    #create reader for each vtk file
    reader1=vtk.vtkGenericDataObjectReader()
    reader1.SetFileName(file)
    reader1.Update()
    
    #get output data
    data = reader1.GetOutput()
    
    #Extract points 
    points=data.GetPoints().GetData()
    points_np=vtk.util.numpy_support.vtk_to_numpy(points)
    
    
    #extract pressure
    p=get_vtk_array(data,'p')
    
   
    #create dataframe for every data point
    df = pd.DataFrame()
    df['x']=points_np[:,0]
    df['y']=points_np[:,1]
    df['z']=points_np[:,2]

    df['p']=p 
    df['Re']=(max(df['x'])-min(df['x']))*30/(1.18*10**(-5))
    
    print("new df created :",df.head())
    
    #subsample from df
    df = df.sample(self.sample_size,replace="False")
    
    #add to bigger dataframe with other points    
    df_subsampled = pd.concat((df_subsampled,df))
    print("New data point added to the training dataset.", len(df_subsampled)/len(df))
    
    #save training dataset if required
   df_subsampled.to_csv("training_data.csv")
  
   #calculate L_norm
   self.L_norm=max(df['Re'])*1.18*10**(-5)/30
   
   #segregate into input and output
   self.input_points=np.array(df_subsampled[['x','y','z']])
   #print(input_points.shape)
   self.output_points=np.array(df_subsampled['p'])
   
    
  
    
 def Make_validation_data(self):
    
    allfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
    df_validation = pd.DataFrame()
    
    for file in allfiles:
        reader=vtk.vtkGenericDataObjectReader()
        reader.SetFileName(file)
        reader.Update()
        
        #get output data
        data = reader.GetOutput()
        
        #Extract points 
        points=data.GetPoints().GetData()
        points_np=vtk.util.numpy_support.vtk_to_numpy(points)
        
        
        #extract pressure
        p=self.get_vtk_array(data,'p')
        
       
        #create dataframe for every data point
        df = pd.DataFrame()
        df['x']=points_np[:,0]
        df['y']=points_np[:,1]
        df['z']=points_np[:,2]

        df['p']=p 
        df['Re']=(max(df['x'])-min(df['x']))*30/(1.18*10**(-5))
        
        print("new df created :",df.head()," ", len(df))
        
        
        
        #add to bigger dataframe with other points    
        df_valiation = pd.concat((df_validation,df))
        print("New data point added to the training dataset.", len(df_valiation)/len(df))
        
        #save training dataset if required
        df.to_csv("val_data.csv")
      
         
    #segregate into input and output
    input_points=np.array(df_validation[['x','y','z']])
    print(input_points.shape)
    output_points=np.array(df_validation['p'])
    
    return input_points, output_points

#create normalization
 def normalization_physics(self):
    
    self.input_points[:,0:4]=self.input_points[:,0:4]/self.L_norm
    self.output_points=self.output_points/(0.5*1.18*self.V_norm**2)
    
    



    
class Data_4_training(torch.utils.data.Dataset):
    
    def __init__(self,input_data,output_data):
        self.input_data=input_data
        self.output_data=output_data
    
    def __len__(self):
        return len(self.input_data)
        
    def __getitem__(self, idx):
        return self.input_data[idx],self.output_data[idx]
    
    
        
        
    
    
    
    
    
    
    
    
#for file in files:
    
    
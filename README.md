# Pressure_predictor_ReGCNN_Drivernet
Modified the ReGCNN workflow from the original authors of Drivernet, to predict pressure distribution and encode/embed geometry for downstream tasks (Proprietary).  Applied on the Drivernet++ surface pressure components. 

Ref to dataset and original code: https://github.com/Mohamedelrefaie/DrivAerNet

The vtk files have been used to modify the  point clouds and convert them into pandas dataframes for operations and subsampling. A custom data pipeline has been used to feed the vtk files to the Dynamic GCNN model. 

Note: Unpublished work for personal practice and learning. This is not affiliated with my employer/tested for industrial deployment. 


Note: to be added, validators,testers and plots 

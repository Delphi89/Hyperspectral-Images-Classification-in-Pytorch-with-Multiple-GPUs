# Hyperspectral Images Classification in Pytorch with Multiple GPUs
Hyperspectral Images Classification in Pytorch - Pavia University

Objective: classification of the hyperspectral images (pavia university map, 103 bands, ~42k pixels)


Notes:
1) The file from generating the database was created in Matlab and is called DB.m
2) At this moment, the maximum classification score is 72%. Work ongoing.
3) this code is using parallel GPU computing for training
4) I'm using Pytorch 0.4, Python 3.6, Anaconda 3, Matlab R2018b, all installed in Windows 10  


DB.m
Description of the Matlab code:
1. original Pavia University database loaded from disk...
2. 2x zero padding added to database (to allow finding neighbours for the pixels from the matrix margins)... 
3. test_data vector (all non-0 values from original file) and test_target vector created using 24 neighbours... 
4. train_data (200 pixels from each class) and train_target created using 24 neighbours...
5. new/clean database saved.

# Hyperspectral Images Classification in Pytorch with Multiple GPUs
Hyperspectral Images Classification in Pytorch - Pavia University

Objective: classification of the hyperspectral images (pavia university map, 103 bands, ~42k pixels)


Notes:
1) The file from generating the database was created in Matlab and is called DB.m
2) At this moment, the maximum classification score is 72%. Work ongoing.
3) this code is using parallel GPU computing for training
4) I'm using Pytorch 0.4, Python 3.6, Anaconda 3, Matlab R2018b, all installed in Windows 10  


DB.m
 - all "0" pixels are eliminated
 - each pixel from the remaining pixels is taken with 24 neighbours and added in a vector
 - from this vector, 200 pixels from each class are taken to be used as a training lot (9 classes x 200 pixels)

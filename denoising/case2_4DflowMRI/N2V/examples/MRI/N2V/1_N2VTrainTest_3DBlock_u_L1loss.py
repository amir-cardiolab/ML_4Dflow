#!/usr/bin/env python
# coding: utf-8

# # N2V Training
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../')

import unet.model
from unet.model import UNet

from pn2v import utils
from pn2v import histNoiseModel
# Loss has been changed to L1 loss in training.py
from pn2v import training
from pn2v import prediction
from tifffile import imread
import time
import os
import torch
# See if we can use a GPU
device=utils.getDevice()


# In[2]:


import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
import sys


##########################################################################
# Function definitions

def read_velocity_data(input_dir, filename, reader, t_1, t_n,velocity_field):
# Read velocity data from file
# Inputs:
# input_dir - input directory location
# filename - velocity timeseries filename 
# reader - vtk reader
# t_1 - first timestep to read
# t_n - last timestep to read
# Outputs:
# X - data matrix containing the velocity data
# mesh - mesh object containing the mesh

    print('Reading velocity data and mesh from:', input_dir + filename, flush = True)

    velocity_list = []
    for i in range(t_1,t_n,1):
        reader.SetFileName(input_dir+filename+str(i)+'.vtu')
        reader.Update()
        output = reader.GetOutput()
        velocity_dataset = output.GetCellData().GetArray(velocity_field)
        velocity = VN.vtk_to_numpy(velocity_dataset)
        velocity_vec = np.reshape(velocity,(-1,1))
        velocity_list.append(velocity_vec)

    # arrange the velocity data into a big data matrix
    X = np.asarray(velocity_list)
    X = X.flatten('F')

    X = np.reshape(X,(-1,t_n-t_1))
    # rows of X correspond to velocity components at spatial locations
    # columns of X correspond to timesteps
    #     t_1 t_2.  .  t_end
    # X = [u  u  .  .  .]  (x_1,y_1)
    #     [v  v  .  .  .]  (x_1,y_1)
    #     [w  w  .  .  .]  (x_1,y_1)
    #     [u  u  .  .  .]  (x_2,y_2)
    #     [v  v  .  .  .]  (x_2,y_2) 
    #     [w  w  .  .  .]  (x_2,y_2)
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .

    # read the mesh for later visualization and saving data
    mesh = reader.GetOutput()

    return X, mesh


# ### Load PIV data data
# 

# Run the cells below

# In[3]:


##########################################################################

#input_dir = "/scratch/hc595/cylinder_flow/CFD_results/navier_stokes_cylinder/downsampled_cropped/"
# input_dir = '/home/hunor/PhD/Phase2/MRI/N2V/data/'
input_dir = '/home/hunor/PhD/Phase2/MRI/MRI_data_vtk/'
# Velocity flag - if True: use velocity data
#                 if False: use vorticity data

filename = 'raw_velocity_'
reader = vtk.vtkUnstructuredGridReader()

t_transient = 0
t_end = 15

# velocity_field = 'u_raw_scaled'
# Xu, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)
# velocity_field = 'v_raw_scaled'
# Xv = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
# velocity_field = 'w_raw_scaled'
# Xw = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]


# In[4]:

# this is done 1-velocity component at a time
filename = 'raw_velocity_'
reader = vtk.vtkUnstructuredGridReader()
velocity_field = 'u'
Xu, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)

# velocity_field = 'v'
# Xv2 = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
# velocity_field = 'w'
# Xw2 = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]


# In[5]:


Xu.shape


# In[6]:


snapshots = t_end
shape_x = 239
shape_y = 192
shape_z = 128
dataName = 'MRI_u'
data = Xu.T
data = data.reshape((-1,shape_y,shape_x),order = 'C')
nameModel=dataName+'_n2v'


# In[7]:


data.shape


# In[8]:


plt.figure(figsize=(15, 15))

plt.title(label='MRI example image')
plt.imshow(data[85,:,:],origin='upper')#, cmap='magma')


# ### Create the Network and Train it
# This can take a while.

# In[ ]:


# The N2V network requires only a single output unit per pixel
net = UNet(1, depth=3)
path = '/home/hunor/PhD/Phase2/MRI/N2V/examples/MRI/N2V/batch1/'
# Split training and validation data.
my_train_data=data[:-220].copy()
my_val_data=data[-220:].copy()

# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix= nameModel, directory=path, noiseModel=None,
                                           device=device, numOfEpochs= 200, stepsPerEpoch=20, 
                                           virtualBatchSize=10, batchSize=1, learningRate=1e-3,
                                           patchSize = 32, numMaskedPixels = 32*32/32.0)


# In[12]:


# Let's look at the training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.semilogy()
plt.plot(valHist, label='validation loss')
plt.plot(trainHist, label='training loss')
plt.legend()
plt.show()


# In[13]:


# Load the network, created in the '01_N2VTraining.ipynb' notebook
net=torch.load(path+"last_"+nameModel+".net")


# In[14]:


# Now we are processing data and calculating PSNR values.
dataTest = data
results=[]
meanRes=[]
resultImgs=[]
inputImgs=[]

# We iterate over all test images.
for index in range(dataTest.shape[0]):
    
    im=dataTest[index]
    
    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    
    means = prediction.tiledPredict(im, net ,ps=64, overlap=10,
                                            device=device, noiseModel=None)
    
    resultImgs.append(means)
    inputImgs.append(im)


    print ("image:",index)
    print ('-----------------------------------')

    
    


# In[15]:


# We display the results for the last test image       
index = 100

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label='Input Image')
plt.imshow(np.asarray(inputImgs)[index,:,:], origin = 'upper')#, vmax=vma, vmin=vmi)#, cmap='magma')

plt.subplot(1, 3, 2)
plt.title(label='Output Image')
plt.imshow(np.asarray(resultImgs)[index,:,:], origin = 'upper')#, vmax=vma, vmin=vmi)#, cmap='magma')



N2Vrecu = np.asarray(resultImgs).reshape((snapshots,-1),order = 'C').T
u_noisy = np.asarray(inputImgs).reshape((snapshots,-1),order = 'C').T


out_filename = 'batch1/N2V_L1_reconstruction_u'
print('Saving the noisy velocity field to ',out_filename)
meshNew = dsa.WrapDataObject(mesh)
#mesh.GetCellData().RemoveArray('u')
for j in range(0,snapshots):
    meshNew.CellData.append(u_noisy[:,j], 'u')
    meshNew.CellData.append(N2Vrecu[:,j],'N2V_u')
#    meshNew.CellData.append(Xv[:,j], 'v')
#    meshNew.CellData.append(Lv[:,j],'v_low_rank_reconstructed')
#    meshNew.CellData.append(Xw[:,j], 'w')
#    meshNew.CellData.append(Lw[:,j],'w_low_rank_reconstructed')
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(out_filename + str(j)+ '.vtk')
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()



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
sys.path.append('../../../') #path to the main folder with Unet and pn2v/utils files!
import unet.model
from unet.model import UNet

from pn2v import utils
from pn2v import histNoiseModel
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

    print('Reading velocity data and mesh from:', input_dir + filename)

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


def convertToMagnitude(X):
# Use velocity magnitude instead of the vector   
# Input:
# X - original data matrix with velocity vector
# Output:
# X_mag - velocity data matrix containing velocity magnitude 
#     t_1   t_2  .  .  t_end
# X_mag = [|u|  |u|  .  .  .]  (x_1,y_1)
#         [|u|  |u|  .  .  .]  (x_2,y_2)
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .

    n = X.shape[0]
    m = X.shape[1]
    X_mag = np.zeros((int(n/3),m))

    for i in range(0,m):
        Ui = X[:,i]
        Ui = np.reshape(Ui,(-1,3))
        Ui_mag = np.sqrt(np.sum(np.square(Ui),1))
        X_mag[:,i] = Ui_mag

    return X_mag


# In[3]:


def add_noise(X, noise_fraction, noise_level):
    n = X.shape[0]
    m = X.shape[1]
    
    #noisy percentage
    n_noise = np.floor(noise_fraction*n*m)
    X_reshaped = np.reshape(X,(n*m,1),order = 'C')
    X_noisy = X_reshaped.copy() 
    
    # create noise at random locations
    rand_interval = np.random.permutation(n*m)
    noise_mask = rand_interval[0:int(n_noise)]
    
    # add Gaussian random noise
    X_noisy[noise_mask] = X_noisy[noise_mask] + noise_level*np.random.randn(X_noisy[noise_mask].size,1)
    
    X_noisy = X_noisy.reshape(n,m, order = 'C')
    
    return X_noisy
    


# ### Load ICA data
# 

# Run the cells below

# In[4]:


############################################################################

input_dir = "/home/sci/hunor.csala/Phase2/ICA/data/Re0/voxelized_cropped/"
filename = 'velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 1000

velocity_field = 'velocity'

X, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)


# In[5]:


convertToMagnitude_flag = False

if convertToMagnitude_flag:
    X = convertToMagnitude(X)
    



# In[6]:


Xu = X[0::3,:]
Xv = X[1::3,:]
Xw = X[2::3,:]


# In[7]:


u=1
l=0
Xmin = np.min(X)
Xmax = np.max(X)
print('X normalized to [0,1], Xmax = ',Xmax,' , Xmin = ',Xmin)



X = (X-Xmin)/(Xmax-Xmin)*(u-l)+l


fraction_data_noise = 0.3
noise_level = 0.1
# add noise
np.random.seed(seed = 42)


# In[8]:


Xn = add_noise(X,fraction_data_noise, noise_level)


# In[9]:


Xu = X[0::3,:]
Xv = X[1::3,:]
Xw = X[2::3,:]

X = np.stack((Xu,Xv,Xw))


# In[10]:


Xu_n = Xn[0::3,:]
Xv_n = Xn[1::3,:]
Xw_n = Xn[2::3,:]

Xn = np.stack((Xu_n, Xv_n, Xw_n))


# In[11]:


c = X.shape[0]
n = X.shape[1]
m = X.shape[2]
print("Data matrix X is c by n by m:",c,"x", n, "x", m)



# In[12]:


snapshots = t_end
shape_x = 30
shape_y = 30
shape_z = 30
dataName = 'ICA_mag'
datau = Xu_n.T
datau = datau.reshape((-1,shape_y,shape_x),order = 'C')
datau_clean = Xu.T.reshape((-1,shape_y,shape_x),order = 'C')

datav = Xv_n.T
datav = datav.reshape((-1,shape_y,shape_x),order = 'C')
datav_clean = Xv.T.reshape((-1,shape_y,shape_x),order = 'C')


dataw = Xw_n.T
dataw = dataw.reshape((-1,shape_y,shape_x),order = 'C')
dataw_clean = Xw.T.reshape((-1,shape_y,shape_x),order = 'C')

nameModel=dataName+'_n2v'



data = np.concatenate((datau,datav,dataw),0)
data_clean = np.concatenate((datau_clean,datav_clean,dataw_clean),0)


# In[13]:
# In[14]:




# ### Create the Network and Train it
# This can take a while.

# In[16]:


# The N2V network requires only a single output unit per pixel
net = UNet(1, depth=5)
path = '/home/sci/hunor.csala/Phase2/ICA/N2V/examples/ICA/N2V/batch8/'
# Split training and validation data.
my_train_data=data[:-100].copy()
my_val_data=data[-100:].copy()


# In[17]:


# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix= nameModel, directory=path, noiseModel=None,
                                           device=device, numOfEpochs= 200, stepsPerEpoch=20, 
                                           virtualBatchSize=10, batchSize=8, learningRate=1e-3,
                                           patchSize = 16, numMaskedPixels = 16*16/16.0)


# In[18]:




# In[43]:


net=torch.load(path+"last_"+nameModel+".net")


# In[44]:


data.shape


# In[45]:


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
    
    means = prediction.tiledPredict(im, net ,ps=16, overlap=4,
                                            device=device, noiseModel=None)
    
    resultImgs.append(means)
    inputImgs.append(im)


    


# In[46]:


# We display the results for the last test image       
index = 14

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label='Input Image')
plt.imshow(np.asarray(inputImgs)[index,:,:], origin = 'lower')#, vmax=vma, vmin=vmi)#, cmap='magma')

plt.subplot(1, 3, 2)
plt.title(label='Output Image')
plt.imshow(np.asarray(resultImgs)[index,:,:], origin = 'lower')#, vmax=vma, vmin=vmi)#, cmap='magma')

plt.subplot(1, 3, 3)
plt.title(label='Clean data')
plt.imshow(data_clean[index,:,:], origin = 'lower')#, vmax=vma, vmin=vmi)#, cmap='magma')



# In[48]:


del datau_clean, datav_clean, dataw_clean, net, datau, datav, dataw 


# In[52]:


N2Vrec = np.asarray(resultImgs).reshape((3*snapshots,-1),order = 'C').T
velo_noisy = np.asarray(inputImgs).reshape((3*snapshots,-1),order = 'C').T


# In[68]:


# reshape ground truth
CFD_data = data_clean.reshape((3*snapshots,-1),order = 'C').T


# In[70]:


abs_error = np.linalg.norm(N2Vrec-CFD_data,'fro')
print("Absolute error: ", abs_error)


# In[75]:


rel_error = np.linalg.norm(N2Vrec-CFD_data,'fro')/np.linalg.norm(CFD_data)
print("Relative error - all: ", rel_error)
rel_error_u = np.linalg.norm(N2Vrec[:,0:snapshots]-CFD_data[:,0:snapshots],'fro')/np.linalg.norm(CFD_data[:,0:snapshots])
print("Relative error - u: ", rel_error_u)
rel_error_v = np.linalg.norm(N2Vrec[:,snapshots:2*snapshots]-CFD_data[:,snapshots:2*snapshots],'fro')/np.linalg.norm(CFD_data[:,snapshots:2*snapshots])
print("Relative error - v: ", rel_error_v)
rel_error_w = np.linalg.norm(N2Vrec[:,2*snapshots:]-CFD_data[:,2*snapshots:],'fro')/np.linalg.norm(CFD_data[:,2*snapshots:])
print("Relative error - w: ", rel_error_w)
avg_error = (rel_error_u + rel_error_v + rel_error_w)/3
print("Relative error - avg:", avg_error)


# In[58]:


out_filename = './batch8/N2V_reconstruction'
print('Saving the noisy velocity field to ',out_filename)
meshNew = dsa.WrapDataObject(mesh)
mesh.GetCellData().RemoveArray('velocity')
for j in range(0,snapshots):
    meshNew.CellData.append(velo_noisy[:,j] * (Xmax - Xmin) + Xmin, 'u')
    meshNew.CellData.append(N2Vrec[:,j] * (Xmax - Xmin) + Xmin,'N2V_u')
    meshNew.CellData.append(velo_noisy[:,snapshots + j] * (Xmax - Xmin) + Xmin, 'v')
    meshNew.CellData.append(N2Vrec[:,snapshots + j] * (Xmax - Xmin) + Xmin,'N2V_v')
    meshNew.CellData.append(velo_noisy[:,2*snapshots + j] * (Xmax - Xmin) + Xmin, 'w')
    meshNew.CellData.append(N2Vrec[:,2*snapshots + j] * (Xmax - Xmin) + Xmin,'N2V_w')
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(out_filename + str(j)+ '.vtk')
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()


# In[ ]:





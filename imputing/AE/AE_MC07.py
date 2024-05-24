#!/usr/bin/env python
# coding: utf-8

# In[1]:


import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torchvision import datasets, transforms


# In[2]:


##########################################################################
# Function definitions

def read_velocity_data(input_dir, filename, reader, t_1, t_n):
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
        # f_18 is the name of the velocity vector dataset assigned by FEniCS for this case
        velocity_dataset = output.GetCellData().GetArray("velocity")
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


def convert3Dto2D_data(X):    
# If the problem is 2D, the w component of the velocity will be all zeros
# These can be deleted to have a smaller data matrix in size
# Input:
# X - velocity data matrix with 3 velocity components
# Output:
# X2D - velocity data matrix with 2 velocity components
#
#       t_1 t_2.  .  t_end
# X2D = [u  u  .  .  .]  (x_1,y_1)
#       [v  v  .  .  .]  (x_1,y_1)
#       [u  u  .  .  .]  (x_2,y_2)
#       [v  v  .  .  .]  (x_2,y_2) 
#       [.  .  .  .  .]   .
#       [.  .  .  .  .]   .
#       [.  .  .  .  .]   . 

    X2D = np.delete(X, list(range(2,X.shape[0],3)),axis = 0)
    return X2D


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



def subtract_mean(X):
# subtract the temporal mean of the data set
# Input:
# X - original data matrix
# Output:
# X - data matrix with temporal mean subtracted
# X_mean - temporal mean of the data
    n = X.shape[0]
    m = X.shape[1]  
    X_mean = np.mean(X,1)
    for i in range(0,n):
        X[i,:] = X[i,:]-X_mean[i]

    X = (1/np.sqrt(m)* X)
    return X, X_mean



def add_missing(snapshot,fraction_missing=0.6):
    n_missing = np.floor(fraction_missing*snapshot.shape[0]*snapshot.shape[1])
    rand_interval = np.random.permutation(snapshot.shape[0]*snapshot.shape[1])
    mask = rand_interval[0:int(n_missing)]
    snapshot = snapshot.cpu()
    snapshot_reshaped = np.reshape(snapshot,(snapshot.shape[0]*snapshot.shape[1],1))
    corrupt = snapshot_reshaped.clone()
    corrupt[mask] = 0.0
    corrupt = corrupt.reshape(snapshot.shape[0],snapshot.shape[1])
    
    return corrupt


# In[5]:


# In[3]:


############################################################################

input_dir = "/home/hunor/PhD/ICA_aneurysm/results/"
filename = 'velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 1000

all_Re = np.arange(8)
X_list = []

convertToMagnitude_flag = True
normalize_flag = True

for rei in all_Re:
    Xi, mesh = read_velocity_data(input_dir +'Re'+str(int(rei)) +'/voxelized_cropped/', filename, reader, t_transient, t_end)
    # convertToMagnitude_flag 
    #                   if True: velocity magnitude will be used |u|
    #                   if False: velocity vector will be used [u v]


    if convertToMagnitude_flag:
        Xi = convertToMagnitude(Xi)
        
        
    #normalize everything to [0,1]
    if normalize_flag:
        u=1
        l=0
        Xi = (Xi-np.min(Xi))/(np.max(Xi)-np.min(Xi))*(u-l)+l

    X_list.append(Xi)

    
X = np.asarray(X_list)


# In[6]:


# In[4]:


subtract_mean_flag = False


if subtract_mean_flag:
    X, X_mean = subtract_mean(X)


total_Re_s = X.shape[0]
n = X.shape[1]
m = X.shape[2]
print("Data matrix X is n by m:", n, "x", m)
print("Total number of different Re:",total_Re_s)


X_train = X[[1,2,3,4,5,6,7],:,:] 
X_test = X[[0],:,:]
print("Split to ", X_train.shape[0]/total_Re_s, "% training and ", X_test.shape[0]/total_Re_s, "% testing data")


# In[5]:


# In[7]:


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
else: print('Running on CPU')


# In[8]:


torch.cuda.is_available()


# In[9]:


#Reshape data to useful form. 
#Simulations at different Re are stacked in row-wise blocks: X= |Re1|Re2|Re3|Re4|...
#Each Re block has the shape:
#          t_1   t_2  .  .  t_end
# Re1  =  [|u|  |u|  .  .  .]  (x_1,y_1)
#         [|u|  |u|  .  .  .]  (x_2,y_2)
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
X_train = X_train.transpose(1,0,2).reshape((n,X_train.shape[0]*m))
X_test = X_test.transpose(1,0,2).reshape((n,X_test.shape[0]*m))


# In[10]:


# Prepare dataset for pyTorch
X_tensor = torch.from_numpy(X_train.T)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = 128
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
index_list = torch.randperm(len(dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)


# In[6]:


fraction_missing = 0.7
n_missing = np.floor(fraction_missing*n*m)
X_reshaped = np.reshape(X_test,(n*m,1))
X_corrupt = X_reshaped.copy()
np.random.seed(42)
rand_interval = np.random.permutation(n*m)
mask = rand_interval[0:int(n_missing)]
X_corrupt[mask] = 0.0
X_corrupt = X_corrupt.reshape(n,m)


# In[7]:


# Prepare dataset for pyTorch
X_tensor_test_corrupt = torch.from_numpy(X_corrupt.T)
X_tensor_test = torch.from_numpy(X_test.T)
dataset_test = torch.utils.data.TensorDataset(X_tensor_test, X_tensor_test_corrupt)
#shuffle data manually and save indices
index_list_test = torch.randperm(len(dataset_test)).tolist()
shuffled_dataset_test = torch.utils.data.Subset(dataset_test, index_list_test)
data_loader_test = torch.utils.data.DataLoader(shuffled_dataset_test, batch_size = batchsize, shuffle = False)


# In[8]:


# In[11]:


# Define autoencoder network structure
class Autoencoder_Linear(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n,8192),
            nn.ReLU(),
            nn.Linear(8192,1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,8)
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,64), 
            nn.ReLU(),
            nn.Linear(64,256), 
            nn.ReLU(),
            nn.Linear(256,1024),
            nn.ReLU(),
            nn.Linear(1024,8192),
            nn.ReLU(),
            nn.Linear(8192,n)
        ) 
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[12]:


# Define loss and optimiziation parameters
model = Autoencoder_Linear().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(),lr = 1e-3, weight_decay = 0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
scheduler_active_flag = True


# In[9]:


# In[13]:

num_epochs = 300
outputs = []
loss_list = []
loss_test_list = []
outputs_test = []
start = time.time()
for epoch in range(num_epochs):
    batch_iter = 0
    loss_tot = 0.0
    loss_tot_test = 0.0
    # Training loop
    for x in data_loader:
        # x is a list originally, so we have to get the first element which is the tensor
        snapshot = x[0].type(torch.FloatTensor).to(device)
        snapshot_noisy = add_missing(snapshot,fraction_missing).to(device)
        
        recon = model(snapshot_noisy)
        loss = criterion(recon, snapshot)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        
        recon, snapshot, snapshot_noisy = recon.detach().cpu(), snapshot.detach().cpu(), snapshot_noisy.detach().cpu()
        
        del recon, snapshot, snapshot_noisy
        
        batch_iter += 1
    loss_tot = loss_tot/batch_iter
    loss_list.append((epoch, loss_tot))
    print(f'Epoch: {epoch+1}, Total avg train loss: {loss_tot:.10f}')
    
    
    #Testing loop
    batch_iter = 0
    for x,y in data_loader_test:
        model.eval()
        with torch.no_grad(): # No need to track the gradients since it's testing
        # x is a list originally, so we have to get the first element which is the tensor
            snapshot = x.type(torch.FloatTensor).to(device)
            snapshot_noisy = y.type(torch.FloatTensor).to(device)

            recon = model(snapshot_noisy)

            loss_test = criterion(recon, snapshot)

            loss_tot_test += loss_test.item()
            
            recon, snapshot, snapshot_noisy = recon.detach().cpu(), snapshot.detach().cpu(), snapshot_noisy.detach().cpu()
            if epoch == num_epochs-1:
                outputs_test.append((epoch+1,batch_iter, snapshot, recon, snapshot_noisy))
            
            del recon, snapshot, snapshot_noisy
            
            batch_iter += 1
    loss_tot_test = loss_tot_test/batch_iter
    loss_test_list.append((epoch, loss_tot_test))
    print(f'Epoch: {epoch+1}, Total avg test loss: {loss_tot_test:.10f}', flush = True)
    
    
    if (scheduler_active_flag):
        scheduler.step()


end = time.time()
print('Time elapsed for training DAE:',end - start)


# In[12]:


# Save testing results

# Organize results for saving and visualization
# Unshuffle results and reconstructions
outx_shuffled = []
outxnoisy_shuffled = []
outxrec_shuffled = []
for i in range(int(np.ceil(X_test.shape[1]/batchsize))):
    outx_shuffled.append(outputs_test[i][2])
    outxrec_shuffled.append(outputs_test[i][3])
    outxnoisy_shuffled.append(outputs_test[i][4])
    
del outputs_test

x_out_shuffled = torch.cat(outx_shuffled).detach().cpu().numpy()
xnoisy_out_shuffled = torch.cat(outxnoisy_shuffled).detach().cpu().numpy()
xrec_out_shuffled = torch.cat(outxrec_shuffled).detach().cpu().numpy()

x_out = np.zeros(x_out_shuffled.shape)
xnoisy_out = np.zeros(xnoisy_out_shuffled.shape)
xrec_out = np.zeros(xrec_out_shuffled.shape)

j = 0
for i in index_list_test:
    x_out[i,:] = x_out_shuffled[j,:]
    xrec_out[i,:] = xrec_out_shuffled[j,:]
    xnoisy_out[i,:] = xnoisy_out_shuffled[j,:]
    j +=1
    
error_rec = np.linalg.norm(x_out-xrec_out,'fro')
print('Testing reconstruction error: %.5e' % (error_rec), flush = True)


# In[13]:


# Save the modes and the reconstructed fieldb
save_rec_flag = True

if(save_rec_flag):
    out_filename = 'Reconstruction/reconstruction_test_AE_'
    print('Saving the reconstructed velocity field to ',out_filename)
    meshNew = dsa.WrapDataObject(mesh)
    mesh.GetCellData().RemoveArray('velocity')
    if convertToMagnitude_flag:
        for j in range(0,x_out.shape[0]):
            meshNew.CellData.append(xrec_out[j,:], 'reconstructed')
            meshNew.CellData.append(x_out[j,:], 'clean')
            meshNew.CellData.append(xnoisy_out[j,:], 'noisy')
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    else: ## this part is probably not working correctly
        for j in range(0,int(x_out.shape[0]/2)):
            meshNew.PointData.append(xrec_out[2*j,:], 'u_reconstructed')
            meshNew.PointData.append(x_out[2*j,:], 'u_original')
            meshNew.PointData.append(xrec_out[2*j+1,:], 'v_reconstructed')
            meshNew.PointData.append(x_out[2*j+1,:], 'v_original')
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[19]:


# In[14]:


# Plot loss as a function of the number of epochs
loss_mat = np.asarray(loss_list)
plt.figure(1)
plt.plot(loss_mat[:,0],loss_mat[:,1],linestyle='--')
loss_mat_test = np.asarray(loss_test_list)
plt.figure(1)
plt.plot(loss_mat_test[:,0],loss_mat_test[:,1],linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AE Loss')
plt.semilogy()
plt.tight_layout()
plt.legend(['Train','Test'])
plt.savefig('DAE_loss.png',dpi = 200)


# In[ ]:





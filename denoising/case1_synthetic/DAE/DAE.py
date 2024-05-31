#!/usr/bin/env python
# coding: utf-8

# In[1]:


import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
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

    print('Reading velocity data and mesh from:', input_dir + filename)

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


def RearrangeDataForTranspose(X):
# Reshape data matrix for temporal reduction
# Each row contains both u and v for a given spatial location
# Each two columns contain a snapshot of u and of v
# The rows of the matrix will be taken as different data points and will be compared to each other
# Therefore, it is not fair to comapre u with v, this necessitates this reshaping
# Input:
# X - original data matrix
# Output:
# X_new - new data matrix, arranged as:
# X_new = [u  v  u  v  .]  (x_1,y_1)
#         [u  v  u  v  .]  (x_2,y_2)
#         [u  v  u  v  .]  (x_3,y_3)
#         [u  v  u  v  .]  (x_4,y_4)
#         [.  .  .  .  .]   .
#         [.  .  .  .  .]   .
#         [.  .  .  .  .]   .
#         t1 t1 t2 t2  .

    u = X[0::2,:]
    v = X[1::2,:]

    n = X.shape[0]
    m = X.shape[1]

    X_new = np.zeros((int(n/2),int(m*2)))
    for i in range(m):
        X_new[:,2*i] = u[:,i]
        X_new[:,2*i+1] = v[:,i]

    return X_new

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


# In[3]:


def add_noise(snapshot,noise_factor=0.1, noise_fraction=0.3):
    # Guassian noise at noise_fraction of the points, scaled with the velocity magnitude
    # noise_factor is the standard deviation (sigma^2) for the normal distribution
    n_noisy = np.floor(noise_fraction*snapshot.shape[0]*snapshot.shape[1])
    rand_interval = np.random.permutation(snapshot.shape[0]*snapshot.shape[1])
    noise_mask = rand_interval[0:int(n_noisy)]
    snapshot = snapshot.cpu()
    snapshot_reshaped = np.reshape(snapshot,(snapshot.shape[0]*snapshot.shape[1],1))
    noisy = snapshot_reshaped.clone()
    noisy[noise_mask] = noisy[noise_mask] + noise_factor*torch.randn_like(noisy[noise_mask])
    noisy = noisy.reshape(snapshot.shape[0],snapshot.shape[1])
#     noisy = torch.clip(noisy,min = 0.0)
    
    # Random noise everywhere, not scaled
    #noisy = snapshot+torch.randn_like(snapshot) * noise_factor
    return noisy 


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


# In[4]:


############################################################################

input_dir = "/home/hunor/PhD/ICA_aneurysm/results/"
filename = 'velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 1000

all_Re = np.arange(8)
X_list = []

convertToMagnitude_flag = False
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
        
        if rei == 0:
            Xmin = np.min(Xi)
            Xmax = np.max(Xi)
            Xi = (Xi-Xmin)/(Xmax-Xmin)*(u-l)+l
            print('X_test normalized to [0,1], Xmax = ',Xmax,' , Xmin = ',Xmin)
        else:

            Xi = (Xi-np.min(Xi))/(np.max(Xi)-np.min(Xi))*(u-l)+l
    
    X_list.append(Xi)
    
X = np.asarray(X_list)


# In[5]:


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
del X


# In[6]:


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
else: print('Running on CPU')


# In[7]:


torch.cuda.is_available()


# In[8]:


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


# In[9]:


fraction_data_noise = 0.3
noise_level = 0.1
n_noise = np.floor(fraction_data_noise*n*m)
X_reshaped = np.reshape(X_test,(n*m,1))
X_noisy = X_reshaped.copy()
del X_reshaped
#fix the seed for reproducible results
np.random.seed(seed = 42)
rand_interval = np.random.permutation(n*m)
noise_mask = rand_interval[0:int(n_noise)]
X_noisy[noise_mask] = X_noisy[noise_mask] + noise_level*np.random.randn(X_noisy[noise_mask].size,1)
X_noisy = X_noisy.reshape(n,m)


# In[10]:


shape_x = 30
shape_y = 30
shape_z = 30


# In[11]:


Xu = X_train[0::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)
Xv = X_train[1::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)
Xw = X_train[2::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)

X_train = np.stack((Xu,Xv,Xw)).transpose(1,0,2,3,4)


Xu = X_test[0::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)
Xv = X_test[1::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)
Xw = X_test[2::3,:].reshape(shape_x,shape_y,shape_z,-1).transpose(3,0,1,2)

X_test = np.stack((Xu,Xv,Xw)).transpose(1,0,2,3,4)


# In[12]:


print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)

del Xu, Xv, Xw


# In[13]:


plt.imshow(X_test[150,1,:,:,15].T,origin='lower')
plt.show()


# In[14]:


plt.imshow(X_noisy.reshape(3,30,30,30,-1,order='F').transpose(0,3,2,1,4)[0,:,:,14,0].T,origin='lower')
plt.colorbar()


# In[15]:


X_noisy = X_noisy.reshape(3,shape_x,shape_y,shape_z,-1,order='F').transpose(0,3,2,1,4)
X_noisy = X_noisy.transpose(4,0,1,2,3)



# Prepare dataset for pyTorch
X_tensor = torch.from_numpy(X_train)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = 128
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
index_list = torch.randperm(len(dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)





# Prepare dataset for pyTorch
X_tensor_test = torch.from_numpy(X_test)
X_tensor_noisy = torch.from_numpy(X_noisy)
dataset_test = torch.utils.data.TensorDataset(X_tensor_test, X_tensor_noisy)
#shuffle data manually and save indices
index_list_test = torch.randperm(len(dataset_test)).tolist()
shuffled_dataset_test = torch.utils.data.Subset(dataset_test, index_list_test)
data_loader_test = torch.utils.data.DataLoader(shuffled_dataset_test, batch_size = batchsize, shuffle = False)


# In[16]:


del X_train,X_test


# In[17]:


del X_tensor_test, X_tensor, X_tensor_noisy


# In[18]:


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
            nn.Linear(64,32)
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(32,64), 
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


# In[19]:


class Autoencoder_3DConv(nn.Module):
    def __init__(self):
        super(Autoencoder_3DConv, self).__init__()
        
        # Encoder (3D Convolutional layers)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Decoder (3D Transposed Convolutional layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=0, output_padding=1)

        )
    
    def forward(self, x):
        # Forward pass through encoder
        encoded = self.encoder(x)
        
        # Forward pass through decoder
        decoded = self.decoder(encoded)

        return decoded


# In[20]:


# Define loss and optimiziation parameters
# model = Autoencoder_Linear().to(device)
model = Autoencoder_3DConv().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(),lr = 1e-3)#, weight_decay = 1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
scheduler_active_flag = True


# In[21]:


noise_factor = noise_level
noise_fraction = fraction_data_noise
num_epochs = 600
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
        snapshot_noisy = add_noise(snapshot.reshape(-1,n),noise_factor,noise_fraction)
        snapshot_noisy = snapshot_noisy.reshape(-1,3,shape_x,shape_y,shape_z).to(device)
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
                outputs_test.append((epoch+1,batch_iter, snapshot,recon, snapshot_noisy))
            else:
                del recon, snapshot, snapshot_noisy
            
            batch_iter += 1
    loss_tot_test = loss_tot_test/batch_iter
    loss_test_list.append((epoch, loss_tot_test))
    print(f'Epoch: {epoch+1}, Total avg test loss: {loss_tot_test:.10f}')
    
    
    if (scheduler_active_flag):
        scheduler.step()


end = time.time()
print('Time elapsed for training DAE:',end - start)



torch.save(model.state_dict(),"./clean_DAE_3Dconv_net" + ".pt")


# Save testing results

# Organize results for saving and visualization
# Unshuffle results and reconstructions
outx_shuffled = []
outxnoisy_shuffled = []
outxrec_shuffled = []
for i in range(int(np.ceil(m/batchsize))):
    outx_shuffled.append(outputs_test[i][2])
    outxrec_shuffled.append(outputs_test[i][3])
    outxnoisy_shuffled.append(outputs_test[i][4])
    
    
del outputs_test

x_out_shuffled = torch.cat(outx_shuffled).detach().cpu().numpy().reshape(-1,n, order = 'F')
xnoisy_out_shuffled = torch.cat(outxnoisy_shuffled).detach().cpu().numpy().reshape(-1,n, order = 'F')
xrec_out_shuffled = torch.cat(outxrec_shuffled).detach().cpu().numpy().reshape(-1,n, order = 'F')

del outx_shuffled, outxnoisy_shuffled, outxrec_shuffled

x_out = np.zeros(x_out_shuffled.shape)
xnoisy_out = np.zeros(xnoisy_out_shuffled.shape)
xrec_out = np.zeros(xrec_out_shuffled.shape)

j = 0
for i in index_list_test:
    x_out[i,:] = x_out_shuffled[j,:]
    xrec_out[i,:] = xrec_out_shuffled[j,:]
    xnoisy_out[i,:] = xnoisy_out_shuffled[j,:]
    j +=1


del x_out_shuffled, xrec_out_shuffled, xnoisy_out_shuffled
    
error_rec = np.linalg.norm(x_out-xrec_out,'fro')
print('Testing reconstruction error: %.5e' % (error_rec))


error_rec_rel = np.linalg.norm(x_out-xrec_out,'fro')/np.linalg.norm(x_out)
print('Testing relative reconstruction error: %.5e' % (error_rec_rel))


err_all = np.linalg.norm(x_out-xrec_out)/np.linalg.norm(x_out)
err_u = np.linalg.norm(x_out[:,0::3]-xrec_out[:,0::3])/np.linalg.norm(x_out[:,0::3])
err_v = np.linalg.norm(x_out[:,1::3]-xrec_out[:,1::3])/np.linalg.norm(x_out[:,1::3])
err_w = np.linalg.norm(x_out[:,2::3]-xrec_out[:,2::3])/np.linalg.norm(x_out[:,2::3])

err_avg = (err_u+err_v+err_w)/3
print("Scaled [0,1] -  results")
print("Relative error - all:", err_all)
print("Relative error - u:", err_u)
print("Relative error - v:", err_v)
print("Relative error - w:", err_w)
print("Relative error - avg:", err_avg)


#scale back to original scale [cm/s]

x_out = x_out * (Xmax - Xmin) + Xmin
xrec_out = xrec_out * (Xmax - Xmin) + Xmin
xnoisy_out = xnoisy_out * (Xmax - Xmin) + Xmin



err_all = np.linalg.norm(x_out-xrec_out)/np.linalg.norm(x_out)
err_u = np.linalg.norm(x_out[:,0::3]-xrec_out[:,0::3])/np.linalg.norm(x_out[:,0::3])
err_v = np.linalg.norm(x_out[:,1::3]-xrec_out[:,1::3])/np.linalg.norm(x_out[:,1::3])
err_w = np.linalg.norm(x_out[:,2::3]-xrec_out[:,2::3])/np.linalg.norm(x_out[:,2::3])

err_avg = (err_u+err_v+err_w)/3
print("Real dimensions -  results")
print("Relative error - all:", err_all)
print("Relative error - u:", err_u)
print("Relative error - v:", err_v)
print("Relative error - w:", err_w)
print("Relative error - avg:", err_avg)


# In[26]:


save_rec_flag = True
# # Save the modes and the reconstructed fieldb
if(save_rec_flag):
    out_filename = 'Reconstruction/reconstruction_test_AE_'
    print('Saving the reconstructed velocity field to ',out_filename)
    meshNew = dsa.WrapDataObject(mesh)
    mesh.GetPointData().RemoveArray('velocity')
    if convertToMagnitude_flag:
        for j in range(0,x_out.shape[0]):
            meshNew.CellData.append(xrec_out[j,:], 'reconstructed')
            meshNew.CellData.append(x_out[j,:], 'clean')
            meshNew.CellData.append(xnoisy_out[j,:], 'noisy')
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    else: 
        for j in range(0,xnoisy_out.shape[0]):

            meshNew.CellData.append(xrec_out[j,0::3], 'u_reconstructed')
            meshNew.CellData.append(x_out[j,0::3], 'u_original')
            meshNew.CellData.append(xnoisy_out[j,0::3], 'u_noisy')

            meshNew.CellData.append(xrec_out[j,1::3], 'v_reconstructed')
            meshNew.CellData.append(x_out[j,1::3], 'v_original')
            meshNew.CellData.append(xnoisy_out[j,1::3], 'v_noisy')
            meshNew.CellData.append(xrec_out[j,2::3], 'w_reconstructed')
            meshNew.CellData.append(x_out[j,2::3], 'w_original')
            meshNew.CellData.append(xnoisy_out[j,2::3], 'w_noisy')

            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')

            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[27]:


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





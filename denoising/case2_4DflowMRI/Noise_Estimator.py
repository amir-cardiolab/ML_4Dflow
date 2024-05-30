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
import random
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim


# In[2]:


##########################################################################
# Function definitions

def read_velocity_data(input_dir, filename, reader, t_1, t_n, velocity_field):
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
    
    return noisy 


# In[33]:


############################################################################

input_dir = "/home/hunor/PhD/Phase2/MRI/CFD/unsteady/cases/"
filename = 'aneurysm_celldata_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 100

# all_Re = np.arange(2)
all_Re = [1,2,3,5]
X_list = []

convertToMagnitude_flag = False 
normalize_flag = False

velocity_field = 'velocity'

for rei in all_Re:
    Xi, mesh = read_velocity_data(input_dir +'case'+str(int(rei)) +'/aneurysm-results/', filename, reader,\
                                  t_transient, t_end, velocity_field)
    #Xi = convert3Dto2D_data(Xi)
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


# In[5]:


subtract_mean_flag = False


if subtract_mean_flag:
    X, X_mean = subtract_mean(X)


total_Re_s = X.shape[0]
n = X.shape[1]
m = X.shape[2]
print("Data matrix X is n by m:", n, "x", m)
print("Total number of different Re:",total_Re_s)


X_train = X[[0,2,3],:,:] 
X_test = X[[1],:,:]
# print("Split to ", X_train.shape[0]/total_Re_s, "% training and ", X_test.shape[0]/total_Re_s, "% testing data")


# In[6]:


X_train.shape


# In[31]:


input_dir = "/home/hunor/PhD/Phase2/MRI/MRI_data_vtk/"
filename = 'aneurysm_velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 15

velocity_field = 'u'

X_test_u, mesh = read_velocity_data(input_dir, filename, reader,\
                              t_transient, t_end, velocity_field)
velocity_field = 'v'
X_test_v = read_velocity_data(input_dir, filename, reader,\
                                  t_transient, t_end, velocity_field)[0]
velocity_field = 'w'
X_test_w = read_velocity_data(input_dir, filename, reader,\
                                  t_transient, t_end, velocity_field)[0]



# In[8]:


X_MRI = np.zeros((n,t_end-t_transient))
#scale up MRI experimental data by 6 to match CFD scale!
X_MRI[0::3] = X_test_u*6
X_MRI[1::3] = X_test_v*6
X_MRI[2::3] = X_test_w*6


# In[9]:


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
else: print('Running on CPU')


# In[10]:


torch.cuda.is_available()


# In[11]:


#Reshape data to useful form. 
#Simulations at different Re are stacked in row-wise blocks: X= |Re1|Re2|Re3|Re4|...
#Each Re block has the shape:
#          t_1   t_2  .  .  t_end
# Re1  =  [|u|  |u|  .  .  .]  (x_1,y_1)
#         [|u|  |u|  .  .  .]  (x_2,y_2)
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
X_train = X_train.transpose(1,0,2).reshape((n,-1))
print('Training data:', X_train.shape)
X_test = X_test.transpose(1,0,2).reshape((n,-1))
print('Testing data:', X_test.shape)


# In[12]:


# Prepare dataset for pyTorch
X_tensor = torch.from_numpy(X_train.T)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = 1
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
index_list = torch.randperm(len(dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)





# Prepare dataset for pyTorch
X_tensor_test = torch.from_numpy(X_test.T)
dataset_test = torch.utils.data.TensorDataset(X_tensor_test)
#shuffle data manually and save indices
index_list_test = torch.randperm(len(dataset_test)).tolist()
shuffled_dataset_test = torch.utils.data.Subset(dataset_test, index_list_test)
data_loader_test = torch.utils.data.DataLoader(shuffled_dataset_test, batch_size = batchsize, shuffle = False)


# In[13]:


class NoiseEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0)
        )
        for i in range(num_layers - 1):
            self.encoder.add_module(f"encoder_layer{i}", nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.0)
            ))
            
            # Apply Xavier initialization to the weights of the layer
            init.xavier_uniform_(self.encoder[-1][0].weight)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0)
        )
        for i in range(num_layers - 1):
            self.decoder.add_module(f"decoder_layer{i}", nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.0)
            ))
            
            # Apply Xavier initialization to the weights of the layer
            init.xavier_uniform_(self.decoder[-1][0].weight)

        # Output
        self.output = nn.Linear(hidden_size, 1)
        
        # Apply Xavier initialization to the weights of the output layer
        init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x


# In[14]:


# Define loss and optimiziation parameters
model = NoiseEstimator(X_train.shape[0],hidden_size = 256, num_layers = 3).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(),lr = 1e-4, weight_decay = 0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
scheduler_active_flag = True


# In[38]:


noise_fraction = 0.99
noise_factor_min = 0.01
noise_factor_max = 0.49

num_epochs = 300
outputs = []
loss_list = []
loss_test_list = []
outputs_test = []
start = time.time()
best_val_loss = 1e10
for epoch in range(num_epochs):
    batch_iter = 0
    loss_tot = 0.0
    loss_tot_test = 0.0
    # Training loop
    model.train()
    for x in data_loader:
        # x is a list originally, so we have to get the first element which is the tensor
        snapshot = x[0].type(torch.FloatTensor).to(device)
        noise_factor = torch.rand(size=(1,1)) * noise_factor_max + noise_factor_min
        snapshot_noisy = add_noise(snapshot,noise_factor,noise_fraction).to(device)
        
        noise_factor_tensor = torch.ones((snapshot.shape[0],1))*noise_factor
        noise_factor_tensor = noise_factor_tensor.to(device)
        recon = model(snapshot_noisy)
        loss = criterion(recon, noise_factor_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        
        recon, snapshot, snapshot_noisy = recon.detach().cpu(), snapshot.detach().cpu(), snapshot_noisy.detach().cpu()
        if epoch == num_epochs-1:
            outputs.append((epoch+1,batch_iter, snapshot, recon, snapshot_noisy))
        
        del snapshot, snapshot_noisy
        
        batch_iter += 1
    loss_tot = loss_tot/batch_iter
    loss_list.append((epoch, loss_tot))
    print(f'Epoch: {epoch+1}, Total avg train loss: {loss_tot:.10f}')
    
    
    #Testing loop
    batch_iter = 0
    for x in data_loader_test:
        model.eval()
        with torch.no_grad(): # No need to track the gradients since it's testing
        # x is a list originally, so we have to get the first element which is the tensor
            snapshot = x[0].type(torch.FloatTensor).to(device)
            snapshot_noisy = x[0].type(torch.FloatTensor).to(device)
            noise_factor = torch.rand(size=(1,1)) * noise_factor_max + noise_factor_min
            snapshot_noisy = add_noise(snapshot,noise_factor,noise_fraction).to(device)
            
            
            noise_factor_tensor = torch.ones((snapshot.shape[0],1))*noise_factor
            noise_factor_tensor = noise_factor_tensor.to(device)
            recon = model(snapshot_noisy)
            
            loss_test = criterion(recon, noise_factor_tensor)



            loss_tot_test += loss_test.item()
            
            recon, snapshot_noisy = recon.detach().cpu(), snapshot_noisy.detach().cpu()
            if epoch == num_epochs-1:
                outputs_test.append((epoch+1,batch_iter, recon, snapshot_noisy))
            
            del snapshot, snapshot_noisy
            
            batch_iter += 1
    loss_tot_test = loss_tot_test/batch_iter
    loss_test_list.append((epoch, loss_tot_test))
    print(f'Epoch: {epoch+1}, Total avg val loss: {loss_tot_test:.10f}')
    if loss_tot_test < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saving best model.....')
            best_val_loss = loss_tot_test
            print('Real noise factor:', noise_factor.cpu().numpy())
            print('Estimated noise factor:', recon.detach().cpu().numpy())
    
    if (scheduler_active_flag):
        scheduler.step()


end = time.time()
print('Time elapsed for training DAE:',end - start)


# In[40]:


# Save training results

# Organize results for saving and visualization
# Unshuffle results and reconstructions
outxrec_shuffled = []
for i in range(int(np.ceil(X_train.shape[1]/batchsize))):
    outxrec_shuffled.append(outputs[i][4])


# In[42]:

xrec_out_shuffled = torch.cat(outxrec_shuffled).detach().cpu().numpy()

xrec_out = np.zeros(xrec_out_shuffled.shape)
j = 0
for i in index_list:
    xrec_out[i,:] = xrec_out_shuffled[j,:]
    j +=1

   


# In[45]:

save_rec_flag = True

if(save_rec_flag):
    out_filename = 'Reconstruction_NoiseEstimator/reconstruction_AE_'
    print('Saving the reconstructed velocity field to ',out_filename)
    meshNew = dsa.WrapDataObject(mesh)
    meshNew.GetCellData().RemoveArray('u')
    meshNew.GetCellData().RemoveArray('v')
    meshNew.GetCellData().RemoveArray('w')
    meshNew.GetCellData().RemoveArray('ufilt')
    meshNew.GetCellData().RemoveArray('vfilt')
    meshNew.GetCellData().RemoveArray('wfilt')
    meshNew.GetCellData().RemoveArray('velocity')
    meshNew.GetCellData().RemoveArray('velocity')
    if convertToMagnitude_flag:
        for j in range(0,x_out.shape[0]):
            meshNew.CellData.append(xrec_out[j,:], 'reconstructed')
            meshNew.CellData.append(x_out[j,:], 'clean')
            meshNew.CellData.append(xnoisy_out[j,:], 'noisy')
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    else: 
        for j in range(0,int(xrec_out.shape[0])):
            meshNew.CellData.append(xrec_out[j,0::3], 'u_reconstructed')
            meshNew.CellData.append(xrec_out[j,1::3], 'v_reconstructed')
            meshNew.CellData.append(xrec_out[j,2::3], 'w_reconstructed')
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtu')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[16]:


# Plot loss as a function of the number of epochs
loss_mat = np.asarray(loss_list)
plt.figure(1)
plt.plot(loss_mat[:,0],loss_mat[:,1],linestyle='--')
loss_mat_test = np.asarray(loss_test_list)
plt.plot(loss_mat_test[:,0],loss_mat_test[:,1],linestyle='-')
plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AE Loss')
plt.semilogy()
plt.tight_layout()
plt.legend(['Train','Test'])
plt.savefig('DAE_loss.png',dpi = 200)




# In[17]:


test_model = NoiseEstimator(X_train.shape[0],hidden_size = 256, num_layers = 3).to(device)
test_model.load_state_dict(torch.load('./best_model.pth'))
test_model.to(device)


# ## Inference on MRI data ##

# In[48]:


X_MRI_tensor = torch.from_numpy(X_MRI)


# In[49]:


infered_noise = np.zeros((X_MRI.shape[1]))
with torch.no_grad():
    for i in range(X_MRI_tensor.shape[1]):
        snapshot = X_MRI_tensor[:,i].to(device)
        snapshot = snapshot.float()
        infered_noise[i] = test_model(snapshot).detach().cpu().numpy()
        print('Snapshot',i,'inferred noise level:', infered_noise[i])
    print('Mean noise level:',np.mean(infered_noise))


# In[ ]:





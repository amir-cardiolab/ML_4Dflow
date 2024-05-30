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
#from skimage.metrics import structural_similarity as ssim


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
    
    noisy[snapshot == 0.0] = 0.0
     
    return noisy 

#calculate PSNR
def PSNR(noisy_data, denoised_data, max_val=1.0):
    mse = torch.mean(torch.square(denoised_data - noisy_data))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


# In[4]:


############################################################################

#input_dir = "/scratch/hc595/cylinder_flow/CFD_results/navier_stokes_cylinder_more_timesteps/downsampled_cropped/"
input_dir = "/home/sci/hunor.csala/Phase2/MRI/data/CFD/"
filename = 'box_aneurysm_celldata_'
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
    Xi, mesh = read_velocity_data(input_dir +'case'+str(int(rei))+'/', filename, reader,\
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
print("Total number of different Re:",total_Re_s, flush = True)


X_train = X[[0,1,3],:,:] 
X_test = X[[2],:,:]
print("Split to ", X_train.shape[0]/total_Re_s, "% training and ", X_test.shape[0]/total_Re_s, "% testing data")

del X


# In[7]:


input_dir = "/home/sci/hunor.csala/Phase2/MRI/data/MRI/"
filename = 'box_aneurysm_velocity_'
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


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
else: print('Running on CPU')


# In[9]:


torch.cuda.is_available()


# In[10]:


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
print('Test data:', X_test.shape)


# In[11]:


# Prepare dataset for pyTorch
X_tensor = torch.from_numpy(X_train.T)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = 16
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
index_list = torch.randperm(len(dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)

del X_train



# Prepare dataset for pyTorch
X_tensor_test = torch.from_numpy(X_test.T)
dataset_test = torch.utils.data.TensorDataset(X_tensor_test)
#shuffle data manually and save indices
index_list_test = torch.randperm(len(dataset_test)).tolist()
shuffled_dataset_test = torch.utils.data.Subset(dataset_test, index_list_test)
data_loader_test = torch.utils.data.DataLoader(shuffled_dataset_test, batch_size = batchsize, shuffle = False)


del X_test
# In[12]:


# Define autoencoder network structure
class Autoencoder_Linear(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n,1024),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(64,32)
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(32,64), 
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(64,256), 
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(1024,n)
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[13]:


# Define loss and optimiziation parameters
model = Autoencoder_Linear().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adamax(model.parameters(),lr = 1e-3)#, weight_decay = 1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
scheduler_active_flag = True


# In[35]:


noise_factor = 0.19
noise_fraction = 1.0
num_epochs = 600
outputs = []
loss_list = []
loss_test_list = []
outputs_test = []
best_val_loss = 1e10

start = time.time()
for epoch in range(num_epochs):
    batch_iter = 0
    loss_tot = 0.0
    loss_tot_test = 0.0
    # Training loop
    for x in data_loader:
        # x is a list originally, so we have to get the first element which is the tensor
        snapshot = x[0].type(torch.FloatTensor).to(device)
        snapshot_noisy = add_noise(snapshot,noise_factor,noise_fraction).to(device)
        snapshot_noisy_2 = add_noise(snapshot,noise_factor,noise_fraction).to(device)
        
#        recon = snapshot_noisy - model(snapshot_noisy)

#        recon2 = snapshot_noisy_2 - model(snapshot_noisy_2)


        recon = model(snapshot_noisy)
        recon2 = model(snapshot_noisy_2)

        loss = 1/2*(criterion(recon, snapshot_noisy_2) + criterion(recon2,snapshot_noisy))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        
        recon, snapshot, snapshot_noisy = recon.detach().cpu(), snapshot.detach().cpu(), snapshot_noisy.detach().cpu()
        if epoch == num_epochs-1:
            outputs.append((epoch+1,batch_iter, snapshot, recon, snapshot_noisy))
        
        del recon, snapshot, snapshot_noisy
        
        batch_iter += 1
    loss_tot = loss_tot/batch_iter
    loss_list.append((epoch, loss_tot))
    print(f'Epoch: {epoch+1}, Total avg train loss: {loss_tot:.10f}', flush = True)
    
    
    #Testing loop
    batch_iter = 0
    for x in data_loader_test:
        model.eval()
        with torch.no_grad(): # No need to track the gradients since it's testing
        # x is a list originally, so we have to get the first element which is the tensor
            snapshot = x[0].type(torch.FloatTensor).to(device)
            snapshot_noisy = x[0].type(torch.FloatTensor).to(device)
            snapshot_noisy = add_noise(snapshot,noise_factor,noise_fraction).to(device)
            snapshot_noisy_2 = add_noise(snapshot,noise_factor,noise_fraction).to(device)

            #recon = snapshot_noisy - model(snapshot_noisy)

            #recon2 = snapshot_noisy_2 - model(snapshot_noisy_2)

            recon = model(snapshot_noisy)
            recon2 = model(snapshot_noisy_2)
	
            loss_test = 1/2*(criterion(recon, snapshot_noisy_2) + criterion(recon2,snapshot_noisy))
            #loss_test = criterion(recon, snapshot_noisy_2)

            loss_tot_test += loss_test.item()
            
            
#             psnr_val = PSNR(snapshot_noisy, recon, recon.max()).detach().cpu().numpy()
            
#             print('PSNR:', psnr_val)
            
            recon, snapshot_noisy, snapshot = recon.detach().cpu(), snapshot_noisy.detach().cpu(), \
                                        snapshot.detach().cpu()
            if epoch == num_epochs-1:
                outputs_test.append((epoch+1,batch_iter,snapshot, recon, snapshot_noisy))
            

            del recon, snapshot_noisy, snapshot
            
            batch_iter += 1
    loss_tot_test = loss_tot_test/batch_iter
    loss_test_list.append((epoch, loss_tot_test))
    print(f'Epoch: {epoch+1}, Total avg val loss: {loss_tot_test:.10f}')
    if loss_tot_test < best_val_loss:
        best_val_loss = loss_tot_test
        print('Best validation loss, Saving model ...')
        torch.save(model.state_dict(), 'best_DAE_model.pth')
    
    if (scheduler_active_flag):
        scheduler.step()


end = time.time()
print('Time elapsed for training DAE:',end - start)


# In[30]:


# Save training results

# Organize results for saving and visualization
# Unshuffle results and reconstructions
outx_shuffled = []
outxnoisy_shuffled = []
outxrec_shuffled = []
for i in range(int(np.ceil(m*3/batchsize))):
    outx_shuffled.append(outputs[i][2])
    outxrec_shuffled.append(outputs[i][3])
    outxnoisy_shuffled.append(outputs[i][4])
del outputs


# In[31]:


x_out_shuffled = torch.cat(outx_shuffled).detach().cpu().numpy()
xnoisy_out_shuffled = torch.cat(outxnoisy_shuffled).detach().cpu().numpy()
xrec_out_shuffled = torch.cat(outxrec_shuffled).detach().cpu().numpy()

x_out = np.zeros(x_out_shuffled.shape)
xnoisy_out = np.zeros(xnoisy_out_shuffled.shape)
xrec_out = np.zeros(xrec_out_shuffled.shape)
j = 0
for i in index_list:
    x_out[i,:] = x_out_shuffled[j,:]
    xrec_out[i,:] = xrec_out_shuffled[j,:]
    xnoisy_out[i,:] = xnoisy_out_shuffled[j,:]
    j +=1

   


# In[34]:


error_rec = np.linalg.norm(x_out-xrec_out,'fro')/np.linalg.norm(x_out)
print('Training relative reconstruction error: %.5e' % (error_rec))


# In[18]:


# torch.save(model.state_dict(),"./DAE_net" + ".pt")

# Save the modes and the reconstructed field
save_rec_flag = True

if(save_rec_flag):
    out_filename = 'Reconstruction/reconstruction_AE_'
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
        for j in range(0,int(xnoisy_out.shape[0])):
            meshNew.CellData.append(x_out[j,0::3], 'u_clean')
            meshNew.CellData.append(x_out[j,1::3], 'v_clean')
            meshNew.CellData.append(x_out[j,2::3], 'w_clean')
            meshNew.CellData.append(xrec_out[j,0::3], 'u_reconstructed')
            meshNew.CellData.append(xrec_out[j,1::3], 'v_reconstructed')
            meshNew.CellData.append(xrec_out[j,2::3], 'w_reconstructed')
            meshNew.CellData.append(xnoisy_out[j,0::3], 'u_noisy')
            meshNew.CellData.append(xnoisy_out[j,1::3], 'v_noisy')
            meshNew.CellData.append(xnoisy_out[j,2::3], 'w_noisy')
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtu')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[37]:


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


error_rec = np.linalg.norm(x_out-xrec_out,'fro')/np.linalg.norm(x_out, 'fro')
print('Relative testing reconstruction error: %.5e' % (error_rec))


# In[42]:


error_rec_u = np.linalg.norm(x_out[:,0:int(n/3)]-xrec_out[:,0:int(n/3)],'fro')/np.linalg.norm(x_out[:,0:int(n/3)], 'fro')


# In[22]:


# Save the modes and the reconstructed fieldb
save_rec_flag = True

if(save_rec_flag):
    out_filename = 'Reconstruction/reconstruction_test_AE_'
    print('Saving the reconstructed velocity field to ',out_filename)
    meshNew = dsa.WrapDataObject(mesh)
    meshNew.GetCellData().RemoveArray('u')
    meshNew.GetCellData().RemoveArray('v')
    meshNew.GetCellData().RemoveArray('w')
    meshNew.GetCellData().RemoveArray('ufilt')
    meshNew.GetCellData().RemoveArray('vfilt')
    meshNew.GetCellData().RemoveArray('wfilt')
    meshNew.GetCellData().RemoveArray('velocity')
    meshNew.GetCellData().RemoveArray('u_clean')
    meshNew.GetCellData().RemoveArray('v_clean')
    meshNew.GetCellData().RemoveArray('w_clean')
#     reconstructed_array = VN.numpy_to_vtk(xrec_out[:,0:int(xnoisy_out.shape[1]/3)])
#     reconstructed_array.SetName("u_reconstructed")
    if convertToMagnitude_flag:
        for j in range(0,xnoisy_out.shape[0]):
            meshNew.CellData.append(xrec_out[j,:], 'reconstructed')
            meshNew.CellData.append(xnoisy_out[j,:], 'noisy')
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    else:
        for j in range(0,xnoisy_out.shape[0]):
            meshNew.CellData.append(x_out[j,0::3], 'u_clean')
            meshNew.CellData.append(x_out[j,1::3], 'v_clean')
            meshNew.CellData.append(x_out[j,2::3], 'w_clean')
            meshNew.CellData.append(xrec_out[j,0::3], 'u_reconstructed')
            meshNew.CellData.append(xrec_out[j,1::3], 'v_reconstructed')
            meshNew.CellData.append(xrec_out[j,2::3], 'w_reconstructed')
            meshNew.CellData.append(xnoisy_out[j,0::3], 'u_noisy')
            meshNew.CellData.append(xnoisy_out[j,1::3], 'v_noisy')
            meshNew.CellData.append(xnoisy_out[j,2::3], 'w_noisy')
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtu')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[23]:


# Plot loss as a function of the number of epochs
loss_mat = np.asarray(loss_list)
plt.subplot(2,1,1)
plt.plot(loss_mat[:,0],loss_mat[:,1],linestyle='--')
loss_mat_test = np.asarray(loss_test_list)
plt.plot(loss_mat_test[:,0],loss_mat_test[:,1],linestyle='-')
plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AE Loss')
plt.semilogy()
plt.tight_layout()
plt.legend(['Train','Validation'])
plt.savefig('DAE_loss.png',dpi = 200)


# ### Inference on MRI data ###

# In[44]:


X_MRI = np.zeros((n,t_end-t_transient))
#scale up MRI experimental data by 6 to match CFD scale!
X_MRI[0::3] = X_test_u*6
X_MRI[1::3] = X_test_v*6
X_MRI[2::3] = X_test_w*6


# In[45]:


#create another instance of the model
test_model = Autoencoder_Linear().to(device)
#load the saved model weights and biases
test_model.load_state_dict(torch.load('./best_DAE_model.pth'))
test_model.to(device)


# In[46]:


recon_MRI = torch.zeros((X_MRI.shape))
X_MRI_tensor = torch.from_numpy(X_MRI)
with torch.no_grad():
    for i in range(X_MRI_tensor.shape[1]):
        snapshot = X_MRI_tensor[:,i].to(device)
        snapshot = snapshot.float()
        recon_MRI[:,i] = test_model(snapshot).detach().cpu()
#         psnr = PSNR(snapshot.detach().cpu(),recon_MRI[:,i], max_val = recon_MRI[:,i].max())
#         print('Snapshot',i,'denoised, PSNR:', psnr)
recon = recon_MRI.detach().cpu().numpy().T


# In[47]:


# Save the modes and the reconstructed fieldb
save_rec_flag = True

if(save_rec_flag):
    out_filename = 'Reconstruction/reconstruction_MRI_AE_'
    print('Saving the reconstructed velocity field to ',out_filename)
    meshNew = dsa.WrapDataObject(mesh)
    meshNew.GetCellData().RemoveArray('u')
    meshNew.GetCellData().RemoveArray('v')
    meshNew.GetCellData().RemoveArray('w')
    meshNew.GetCellData().RemoveArray('ufilt')
    meshNew.GetCellData().RemoveArray('vfilt')
    meshNew.GetCellData().RemoveArray('wfilt')
    meshNew.GetCellData().RemoveArray('velocity')
    meshNew.GetCellData().RemoveArray('u_clean')
    meshNew.GetCellData().RemoveArray('v_clean')
    meshNew.GetCellData().RemoveArray('w_clean')
    if convertToMagnitude_flag:
        for j in range(0,xnoisy_out.shape[0]):
            meshNew.CellData.append(xrec_out[j,:], 'reconstructed')
            meshNew.CellData.append(xnoisy_out[j,:], 'noisy')
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    else:
        for j in range(0,recon.shape[0]):
            meshNew.CellData.append(recon[j,0::3], 'u_reconstructed')
            meshNew.CellData.append(recon[j,1::3], 'v_reconstructed')
            meshNew.CellData.append(recon[j,2::3], 'w_reconstructed')
            meshNew.CellData.append(X_MRI.T[j,0::3], 'u_noisy')
            meshNew.CellData.append(X_MRI.T[j,1::3], 'v_noisy')
            meshNew.CellData.append(X_MRI.T[j,2::3], 'w_noisy')
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtu')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


# In[48]:
print("Calculating error for denoised MRI")

filename = 'box_aneurysm_velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 15

velocity_field = 'u'
Xu, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)
velocity_field = 'v'
Xv = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
velocity_field = 'w'
Xw = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]

velocity_field = 'ufilt'
Xu_filt = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
velocity_field = 'vfilt'
Xv_filt = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
velocity_field = 'wfilt'
Xw_filt = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]

#Scale raw, filtered, rpca and n2v by 6
print('Scaling up raw, filtered results by 6 to match CFD scale!')

Xu = Xu*6
Xv = Xv*6
Xw = Xw*6

Xu_filt = Xu_filt*6
Xv_filt = Xv_filt*6
Xw_filt = Xw_filt*6

print('Reading "ground-truth" CFD data')

input_dir = '/home/sci/hunor.csala/Phase2/MRI/data/CFD/original/'
# Velocity flag - if True: use velocity data
#                 if False: use vorticity data

# filename = 'aneurysm_celldata_'
filename = 'box_time_interp__'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 15

velocity_field = 'velocity'
X_cfd, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)
#deconstruct velocity vector
Xu_cfd = X_cfd[0::3,:]
Xv_cfd = X_cfd[1::3,:]
Xw_cfd = X_cfd[2::3,:]


print("Read reconstrued 4D Flow MRI data from the current simulation")

input_dir = './Reconstruction/'
# Velocity flag - if True: use velocity data
#                 if False: use vorticity data

filename = 'reconstruction_MRI_AE_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 0
t_end = 15
velocity_field = 'u_reconstructed'
Xu_dae = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
velocity_field = 'v_reconstructed'
Xv_dae = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]
velocity_field = 'w_reconstructed'
Xw_dae = read_velocity_data(input_dir, filename, reader, t_transient, t_end, velocity_field)[0]


cfd_filt_error_u = np.linalg.norm(Xu_cfd-Xu_filt)
cfd_raw_error_u = np.linalg.norm(Xu_cfd-Xu)
cfd_dae_error_u = np.linalg.norm(Xu_cfd-Xu_dae)
# cfd_rpca_error_u = np.linalg.norm(Xu_cfd-Xu_rpca)
# cfd_n2v_error_u = np.linalg.norm(Xu_cfd-Xu_n2v)

cfd_filt_error_v = np.linalg.norm(Xv_cfd-Xv_filt)
cfd_raw_error_v = np.linalg.norm(Xv_cfd-Xv)
cfd_dae_error_v = np.linalg.norm(Xv_cfd-Xv_dae)
# cfd_rpca_error_v = np.linalg.norm(Xv_cfd-Xv_rpca)
# cfd_n2v_error_v = np.linalg.norm(Xv_cfd-Xv_n2v)

cfd_filt_error_w = np.linalg.norm(Xw_cfd-Xw_filt)
cfd_raw_error_w = np.linalg.norm(Xw_cfd-Xw)
cfd_dae_error_w = np.linalg.norm(Xw_cfd-Xw_dae)
# cfd_rpca_error_w = np.linalg.norm(Xw_cfd-Xw_rpca)
# cfd_n2v_error_w = np.linalg.norm(Xw_cfd-Xw_n2v)


# print('u error| cfd-filter: ',cfd_filt_error_u,'cfd-raw: ',cfd_raw_error_u,\
#      'cfd-rpca: ',cfd_rpca_error_u,'cfd-n2v: ',cfd_n2v_error_u)
print('Pointwise norm error: |X_CFD - X_noisy|_F')
print('u error| cfd-filter: {:.2f}, cfd-raw: {:.2f}, cfd-dae: {:.2f} '\
      .format(cfd_filt_error_u, cfd_raw_error_u,cfd_dae_error_u))
#      'cfd-rpca: ',cfd_rpca_error_u)#'cfd-n2v: ',cfd_n2v_error_u)
print('v error| cfd-filter: {:.2f}, cfd-raw: {:.2f}, cfd-dae: {:.2f} '\
      .format(cfd_filt_error_v, cfd_raw_error_v,cfd_dae_error_v))
#      'cfd-rpca: ',cfd_rpca_error_u)#,'cfd-n2v: ',cfd_n2v_error_u)
print('w error| cfd-filter: {:.2f}, cfd-raw: {:.2f}, cfd-dae: {:.2f}'\
      .format(cfd_filt_error_w, cfd_raw_error_w,cfd_dae_error_w))


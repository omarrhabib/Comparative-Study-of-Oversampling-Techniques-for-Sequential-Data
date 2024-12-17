#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

filename="\q2_train.csv"
DATA_DIR = os.path.join(os.path.abspath("../.."), "Data","quartiles")

df = pd.read_csv(DATA_DIR+filename)
df=df[df['finalResult']=='Pass']
df.head()

#### Contains Fail instances of Synthetic Q1 and Pass instances of original Q2

filenameQ1="..\CGAN Q1\Q1_Synthetic.csv"

q1Synth = pd.read_csv(filenameQ1)
q1Synth=q1Synth[q1Synth['finalResult']=='Fail']
q1Synth.head()

labels,a = pd.factorize(q1Synth['codeModule'])
q1Synth.codeModule=labels
labels,a = pd.factorize(q1Synth['moduleSession'])
q1Synth.moduleSession=labels

q1Synth.head()
data=df.append(q1Synth)
data.shape

with open(DATA_DIR+'/meta.pkl', 'rb') as file:  # Overwrites any existing file.
    d=pickle.load(file)
codeModuleMap=d[0]
moduleSessionMap=d[1]

# encode categorical columns for learning
labels,finalResultMap = pd.factorize(data['finalResult'])
data.finalResult=labels
data.head()

classes_count = 2 # Pass,Fail,Withdrawn
epochs=2000
input_size = data.shape[1]
input_size

##GAN###
import torch, time, pickle
import torch.nn as nn
import torch.optim as optim

##helper functions ###

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)
    plt.show()
    plt.close()
    
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # using input feature size to scale random weights (best practice for ReLU AF)
            n = m.in_features
            y = np.sqrt(2.0/n)
            
            m.weight.data.normal_(0, y)
            m.bias.data.zero_()
            
            
            
##### Gen and Disc networks####
class generator(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, class_num):
        super(generator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        #Has 2 Linear FC layers with 23+500 an 512 neurons each and ReLU AF
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)

        return x

class discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, class_num):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        #Has 2 Linear FC layers with 23+500 an 1024 neurons each and ReLU AF
        self.fc = nn.Sequential(
            nn.Linear(input_dim + class_num, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.output_dim),
            nn.Sigmoid(), # output is binary
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)

        return x
    
    
#####CGAN Network ####
class CGAN(object):
    def __init__(self, epoch, batch_size, save_dir, result_dir, log_dir, gpu_mode, input_size, dataloader,z_dim=69,
                model_dir=datetime.now().strftime("%X").replace(':','.') + '_epoch-%d'%epochs,lrG=0.0002, lrD=0.0002,
                beta1=0.5, beta2=0.9999, class_num=classes_count):
        # parameters
        self.save_dir = save_dir
        self.model_dir=model_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.model_name = self.__class__.__name__
        self.gpu_mode = gpu_mode
        self.epoch = epoch
        self.batch_size = batch_size
        self.input_size = input_size
        self.z_dim = z_dim  # noise for generator
        self.class_num = class_num
        self.sample_num = self.class_num ** 2

        # load dataset
        self.data_loader = dataloader # (self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))
        
        print('---------- Networks architecture -------------')
        print_network(self.D)
        print('-----------------------------------------------')

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()  # binary cross entropy loss
        else:
            self.BCE_loss = nn.BCELoss()

        # fixed noise samples 
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        # fixed noise conditions sample
        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

            
        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break  # epoch complete

                z_ = torch.rand((self.batch_size, self.z_dim))
                #places one at specified index location to one-hot encode y
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                
                y_fill_ = y_vec_ 
                if self.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                D_loss=self.updateDiscriminator(z_, x_, y_vec_, y_fill_)

                # update G network
                G_loss=self.updateGenerator(z_, x_, y_vec_, y_fill_)
                G_loss=self.updateGenerator(z_, x_, y_vec_, y_fill_)
                self.train_hist['G_loss'].append(G_loss.item())
                
                if ((iter + 1) % 30) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            
            print("Epoch: [%2d] completed\t D_loss: %.8f, G_loss: %.8f" %(epoch + 1,D_loss.item(), G_loss.item()))
            
            #view data after every 100-epochs and save it as well
            if epoch%100==0 and epoch>0:
                print('Max:\t',np.max(self.generateDataBatch().iloc[:,3:].max(axis=0).values))
                self.save(epoch)
                
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
            if epoch%500==0 and epoch>0:
                loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.model_name)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save(epochs)
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_dir), self.model_name)

    def updateDiscriminator(self, z_, x_, y_vec_, y_fill_):
        self.D_optimizer.zero_grad()
        D_real = self.D(x_, y_fill_)
        D_real_loss = self.BCE_loss(D_real, self.y_real_)

        G_ = self.G(z_, y_vec_)
        D_fake = self.D(G_, y_fill_)
        D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

        D_loss = D_real_loss + D_fake_loss
        self.train_hist['D_loss'].append(D_loss.item())

        D_loss.backward()
        self.D_optimizer.step()
        return D_loss
        
    def updateGenerator(self,z_, x_, y_vec_, y_fill_):
        self.G_optimizer.zero_grad()
        G_ = self.G(z_, y_vec_)
        D_fake = self.D(G_, y_fill_)
        G_loss = self.BCE_loss(D_fake, self.y_real_)
        G_loss.backward()
        self.G_optimizer.step()
        return G_loss
        
    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + '/' + self.model_name)

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        samples = (samples + 1) / 2
        
        if self.gpu_mode:
            samples = samples.cpu().data.numpy()
        else:
            samples = samples.data.numpy()

        np.savetxt(self.result_dir+'/epoch-%d.txt'%epoch, samples, delimiter=',')
        
    def generateDataBatch(self,classLabel=False):
        if classLabel:
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(classLabel, classLabel+1, (self.batch_size, 1)).type(torch.LongTensor), 1)
        else:
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num, (self.batch_size, 1)).type(torch.LongTensor), 1)
        sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()
            samples = self.G(sample_z_, sample_y_)

        samples = (samples + 1) / 2


        if self.gpu_mode:
            samples = samples.cpu().data.numpy()
            sample_y_= sample_y_.cpu().data.numpy()
        else:
            samples = samples.data.numpy()
            sample_y_= sample_y_.data.numpy()

        samples=samples.round()

        synthData=np.c_[sample_y_[:,0],samples]

        synthDataDf=pd.DataFrame(synthData,columns=data.columns)

        synthDataDf.finalResult = list(finalResultMap[i] for i in np.argmax(sample_y_,axis=1))
        synthDataDf.moduleSession = synthDataDf.apply(
            lambda x:'%s' % moduleSessionMap[x['moduleSession'] 
                                             if x['moduleSession']>=0 and x['moduleSession']<len(moduleSessionMap) 
                                             else (
                                                 0 if x['moduleSession']>0 else len(moduleSessionMap)-1
                                             ) ],
            axis=1
        )
        synthDataDf.codeModule = synthDataDf.apply(
            lambda x:'%s' % codeModuleMap[x['codeModule'] 
                                             if x['codeModule']>=0 and x['codeModule']<len(codeModuleMap) 
                                             else (
                                                 0 if x['codeModule']>0 else len(codeModuleMap)-1
                                             ) ],
            axis=1
        )

        return synthDataDf
    
    def generateData(self,batches=1,classLabel=False):
        if classLabel and type(classLabel)== str:
            try:
                classLabel=finalResultMap.index(classLabel)
            except:
                classLabel=2  #default generate for Fail class
        elif classLabel<0 or classLabel>len(finalResultMap)-1:
            classLabel=2  #default generate for Fail class
                
        df=pd.DataFrame(columns=data.columns)
        for b in range(batches):
            df=pd.concat([df,self.generateDataBatch(classLabel=classLabel)])
        return df
        
    def save(self,e):
        save_dir = os.path.join(self.save_dir, self.model_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G_epoch-%d.pkl'%e))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D_epoch-%d.pkl'%e))

        with open(os.path.join(save_dir, self.model_name + '_history_epoch-%d.pkl'%e), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self,e):
        save_dir = os.path.join(self.save_dir, self.model_dir)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G_epoch-%d.pkl'%e)))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D_epoch-%d.pkl'%e)))
        print(os.path.join(save_dir, self.model_name + '_history_epoch-%d.pkl'%e))
        with open(os.path.join(save_dir, self.model_name + '_history_epoch-%d.pkl'%e),'rb') as input_file:
            self.train_hist=pickle.load(input_file)
            
            
####Pytorhc CUstom DataLoader###
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        y_columns = ['finalResult']
        x_columns = list(set(df.columns) - set(y_columns))
        
        self.x = df[x_columns]
        self.y = df[y_columns]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
    
        x = self.x.iloc[idx].values
        y = float(self.y.iloc[idx].values[0])
        
        x = torch.from_numpy(x).type(torch.FloatTensor)
        
        if self.transform:
            x = self.transform(x)
            
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
    
    
batch_size = 512

d = CustomDataset(data)
data_loader = DataLoader(d, batch_size=batch_size, shuffle=True)



def loadAndPlotModel(e,m_dir='02.54.07_epoch-2000'):
    model = CGAN(
        epoch=e,
        batch_size=batch_size,
        save_dir='models',
        result_dir='results',
        log_dir='logs',
        gpu_mode=True,
        input_size=input_size,
        dataloader=data_loader,
        z_dim=z_dim,
        lrG=lrG,
        model_dir=m_dir,
        lrD=lrD,
        beta1=beta1,
        beta2=beta2
    )
    model.load(e)
    loss_plot(model.train_hist, os.path.join(model.save_dir, model.model_dir), model.model_name)
    
    return model


epoch = epochs
z_dim = 22
beta1 = 0.5
beta2 = 0.9
lrG = 0.00012
lrD = 0.00004

cgan = CGAN(
    epoch=epoch,
    batch_size=batch_size,
    save_dir='models',
    result_dir='results',
    log_dir='logs',
    gpu_mode=True,
    input_size=input_size,
    dataloader=data_loader,
    z_dim=z_dim,
    lrG=lrG,
    lrD=lrD,
    beta1=beta1,
    beta2=beta2
)

cgan.train()


cgan.generateDataBatch().head()

synthData=cgan.generateData(batches=50)
synthData.to_csv("./Q2_synthetic.csv", columns=synthData.columns,index=False)


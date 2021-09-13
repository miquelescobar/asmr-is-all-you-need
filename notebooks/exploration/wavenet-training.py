#!/usr/bin/env python
# coding: utf-8

# # WaveNet - Fit a Sample

# In[1]:

import os
os.chdir('/veu4/usuaris26/footanalytics/segmentation/asmr-is-all-you-need/notebooks/exploration')


import sys
#sys.path.append('../../src/')
sys.path.append('../../network/')


# In[2]:


import os
import torch
from types import SimpleNamespace
torch.cuda.empty_cache()


# In[3]:


from models.wavenet.model import WaveNet
from models.wavenet.utils.data import DataLoader


# In[16]:


params = SimpleNamespace(
    layer_size=10,
    stack_size=5,
    in_channels=256,
    res_channels=512,
    lr=2e-3,
    sample_size=50_000,
    sample_rate=16_000,
    epochs=100_000,
    model_dir='../../network/weights/wavenet/'
)


# In[17]:


class Trainer:
    def __init__(self,
                 layer_size: int = 10,
                 stack_size: int = 5,
                 in_channels: int = 256,
                 res_channels: int = 512,
                 lr: float = 2e-3,
                 sample_size: int = 100_000,
                 sample_rate: int =22_050,
                 epochs: int = 10_000,
                 data_dir: str = '.',
                 model_dir: str = './',
                 model_name: str = None):
        """
        """
        self.epochs = epochs
        self.model_dir = model_dir
        self.model_name = model_name
        
        self.wavenet = WaveNet(layer_size, stack_size, in_channels, res_channels, lr=lr)

        self.data_loader = DataLoader(data_dir, self.wavenet.receptive_fields,
                                      sample_size, sample_rate, in_channels)

    def infinite_batch(self):
        while True:
            for dataset in self.data_loader:
                for inputs, targets in dataset:
                    yield inputs, targets

    def run(self):
        total_steps = 0

        for inputs, targets in self.infinite_batch():
            loss = self.wavenet.train(inputs, targets)

            total_steps += 1

            print('[{0}/{1}] loss: {2}'.format(total_steps, self.epochs, loss))

            if total_steps > self.epochs:
                break

            if total_steps % 2_500 == 0:
                self.wavenet.save(self.model_dir, self.model_name)


# In[18]:


params.data_dir = '../../data/processed/whispering/female/partial/'
params.model_name = 'wavenet-whispering-female'


# In[19]:


trainer = Trainer(**params.__dict__)
#trainer.wavenet.load(params.model_dir, params.model_name)
trainer.run()


# In[ ]:





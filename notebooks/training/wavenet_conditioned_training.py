#!/usr/bin/env python
# coding: utf-8

# # WaveNet - Fit a Sample

# In[1]:


import sys
#sys.path.append('../../src/')
sys.path.append('../../network/')


# In[2]:


import os
import torch
from types import SimpleNamespace
torch.cuda.empty_cache()


# In[3]:


from models.wavenet_conditioned.model import WaveNet
from models.wavenet_conditioned.utils.data import DataLoader


# In[4]:


params = SimpleNamespace(
    layer_size=8,
    stack_size=4,
    in_channels=256,
    res_channels=512, # 256,
    lr=2e-3,
    sample_size=32_000, # 5_000,
    sample_rate=16_000,
    epochs=30_000,
    model_dir='../../network/weights/wavenet_conditioned/'
)


# In[5]:


class Trainer:
    def __init__(self,
                 layer_size: int = 10,
                 stack_size: int = 5,
                 in_channels: int = 256,
                 res_channels: int = 512,
                 num_global_classes: int = 10, # should be length of directory
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
            for class_num, dataset in enumerate(self.data_loader):
                for inputs, targets in dataset:
                    yield inputs, targets, class_num

    def run(self):
        total_steps = 0

        for inputs, targets, class_num in self.infinite_batch():
            print(f'{inputs.shape}, targets: {targets.shape}')
            loss = self.wavenet.train(inputs, targets, class_num)

            print('[{0}/{1}] loss: {2}'.format(total_steps, self.epochs, loss))

            if total_steps % 200 == 0:
                self.wavenet.save(self.model_dir, f'{self.model_name}-{total_steps}epoch')
            
            if total_steps > self.epochs:
                break
                
            total_steps += 1

        self.wavenet.save(self.model_dir, self.model_name)


# In[6]:


params.data_dir = '../../data/processed/conditioned_inputs/'
params.num_global_classes = len(os.listdir(params.data_dir))
params.model_name = 'conditioned'


# In[8]:


trainer = Trainer(**params.__dict__)


# In[ ]:


trainer.run()


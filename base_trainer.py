import os
import time
import random
from functools import lru_cache
from torch import optim

from datasets import *
from utils import *
from models import *
from losses import *

class Base_Trainer():
    def __init__(self, args=None):
        self.initialization(args)
        self.net = RRDBNet(nf=64, nb=8)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.seed = random.seed(100)
        self.dst = MegVSR_Dataset(self.root_dir, crop_per_image=self.crop_per_image)
        self.dataloader_train = DataLoader(self.dst, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=self.num_workers)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, gamma=0.5, step_size=self.step_size)
        self.net = self.net.to(self.device)
    
    def initialization(self, args):
        self.device = 'cuda'
        self.hostname, self.root_dir, self.multi_gpu = get_host_with_dir('/MegVSR')
        self.model_dir = "./saved_model"
        self.train_steps = 1000
        self.batch_size = 8
        self.crop_per_image = 8
        self.crop_size = 64
        self.num_workers = 2
        self.step_size = 50
        self.learning_rate = 1e-4
        self.lastepoch = 0
        self.save_freq = 1
        self.plog_freq = 1
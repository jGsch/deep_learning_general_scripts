import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy
import sys
import time
import os
import pretrainedmodels
import pickle
#from loss.cross_entropy_2d import *
from general_scripts.model.linknet import linknet
from general_scripts.metrics import Score



class Model(object):
    """ Model class for Network, Optimizer, Criterion and scores
    """
    def __init__(self, config, comments='', no_save=False):
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.score = None
        self.best_weights = None

        self.n_epoch = config['n_epochs']

         # time _ method _ batch size _ growth_stage _ downscalling _ mixup
        self.save_path = '%s/run/%s_training' \
                         % (config['save_path'], time.strftime('%m.%d_%H:%M:%S'))

        if not no_save:
            os.makedirs(self.save_path)


    def save(self, epoch=None, checkpoint=False):
        """ Save the model
        """

        epoch = self.n_epoch if epoch == None else epoch

        state = {'epoch': epoch,
                 'model': self.net,
                 'model_state': self.net.state_dict(),
                 'model_state_best': self.best_weights,
                 'optimizer': self.optimizer,
                 'optimizer_state': self.optimizer.state_dict(),
                 'scheduler': self.scheduler,
                 'scheduler_state': self.scheduler.state_dict(),
                 'scores': self.score.get_all_scores()}

        if checkpoint:
            path = '%s/model_and_scores_CHECKPOINT_(epoch_'  % self.save_path
            torch.save(state, '%s%d)' % (path, epoch))
            if epoch > 0:
                os.remove('%s%d)' % (path, epoch-1))
        else:
            path = '%s/model_and_scores_FINAL' % self.save_path
            torch.save(state, path)
            print('Model and scores saved at:\n \'%s\'\n\n' % path)


    def load(self, save_path, only_model=False):
        """ Load the model
        """
        if os.path.isfile(save_path):
            checkpoint = torch.load(save_path)

            self.net = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.score.load_all_scores(checkpoint['scores'])
            print(" Checkpoint Loaded (epoch %d)" % (epoch))
            return checkpoint['epoch'], checkpoint['score']
        else:
            print(" No checkpoint found at '%s'" % save_path)
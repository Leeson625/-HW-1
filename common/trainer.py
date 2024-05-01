# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络的训练的类
    """
    def __init__(self, network, x_train, t_train, x_val, t_val,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},  verbose=True , save_model=False):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.save_model = save_model

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []

        self.best_val_accuracy = 0
        self.best_params = {}

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        # loss = self.network.loss(x_batch, t_batch)
        # if self.verbose: 
        #     print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
             
            train_acc = self.network.accuracy(self.x_train, self.t_train)
            train_loss = self.network.loss(self.x_train, self.t_train)
            val_acc = self.network.accuracy(self.x_val, self.t_val)
            val_loss = self.network.loss(self.x_val, self.t_val)

            self.train_acc_list.append(train_acc)
            self.train_loss_list.append(train_loss)
            self.val_acc_list.append(val_acc)
            self.val_loss_list.append(val_loss)

            if self.verbose: 
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", val acc:" + str(val_acc) + " ===")

            if  self.save_model:
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_params = {k: np.copy(v) for k, v in self.network.params.items()}
                    self.network.save_params("best_model.pkl")
                    print("Saved Best Model.")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()


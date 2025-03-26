# SPINE-master\code\model\main.py
import os
import sys
import torch
from torch import nn
import argparse

from .spine_model import SPINEModel
from random import shuffle
import numpy as np
import logging
import numpy as np
import logging
from sklearn.datasets import make_blobs
import json
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
# SPINE-master\code\model\main.py
import numpy as np
import logging
from sklearn.datasets import make_blobs
import json
from pathlib import Path
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
temp_dir = Path(__file__).parent.parent.parent.parent/"util"/"tmp"


class DataHandler:
    def __init__(self):
        pass
    def loadData(self, train_data):
        self.data = train_data
	
      
        self.data = np.array(self.data)
        self.data_size = self.data.shape[0]
        self.inp_dim = self.data.shape[1]
        self.original_data = self.data[:]
        logging.debug("original_data[0][0:5] = " + str(self.original_data[0][0:5]))


    def getDataShape(self):
        return self.data.shape

    def resetDataOrder(self):
        self.data = self.original_data[:]
        logging.debug("original_data[0][0:5] = " + str(self.original_data[0][0:5]))

    def getNumberOfBatches(self, batch_size):
        return int(( self.data_size + batch_size - 1 ) / batch_size)

    def getBatch(self, i, batch_size, noise_level, denoising):
        batch_y = self.data[i*batch_size:min((i+1)*batch_size, self.data_size)]
        batch_x = batch_y
        if denoising:
            batch_x = batch_y + get_noise_features(batch_y.shape[0], self.inp_dim, noise_level)
            return batch_x, batch_y

    def shuffleTrain(self):
        indices = np.arange(self.data_size)
        np.random.shuffle(indices)
        self.data = self.data[indices]

############################################

def compute_sparsity(X):
	non_zeros = 1. * np.count_nonzero(X)
	total = X.size
	sparsity = 100. * (1 - (non_zeros)/total)
	return sparsity

def dump_vectors(X, outfile, words):
	print ("shape", X.shape)
	assert len(X) == len(words) #TODO print error statement
	fw = open(outfile, 'w')
	for i in range(len(words)):
		fw.write(words[i] + " ")
		for j in X[i]:
			fw.write(str(j) + " ")
		fw.write("\n")
	fw.close()

def get_noise_features(n_samples, n_features, noise_amount):
	noise_x,  _ =  make_blobs(n_samples=n_samples, n_features=n_features, 
			cluster_std=noise_amount,
			centers=np.array([np.zeros(n_features)]))
	return noise_x



class Solver:

    def __init__(self,train_data,hparams):
        # Build data handler
        self.data_handler = DataHandler()
        self.data_handler.loadData(train_data)
        self.input_dim = self.data_handler.getDataShape()[1]
        self.hparams = hparams 
        # Build model
        self.model = SPINEModel(self.input_dim,self.hparams)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.float32
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.lr)
   

    def train(self):
        num_epochs, batch_size = self.hparams.epoch,64
        optimizer = self.optimizer
        dtype = self.dtype
        for iteration in range(num_epochs):
            self.model.train()
            self.data_handler.shuffleTrain()
            num_batches = self.data_handler.getNumberOfBatches(batch_size)
            epoch_losses = np.zeros(4)  # rl, asl, psl, total
            for batch_idx in range(num_batches):
                optimizer.zero_grad()
                batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size, self.hparams.noise, True)
                batch_x = torch.from_numpy(batch_x).to(dtype).to(self.device).requires_grad_()
                batch_y = torch.from_numpy(batch_y).to(dtype).to(self.device)
                out, h, loss, loss_terms = self.model(batch_x, batch_y)
                reconstruction_loss, psl_loss, asl_loss = loss_terms
                loss.backward()
                optimizer.step()
                epoch_losses[0] += reconstruction_loss.item()
                epoch_losses[1] += asl_loss.item()
                epoch_losses[2] += psl_loss.item()
                epoch_losses[3] += loss.item()
            logging.info(f"After epoch {iteration + 1}, Reconstruction Loss = {epoch_losses[0]:.4f}, ASL = {epoch_losses[1]:.4f}, PSL = {epoch_losses[2]:.4f}, and total = {epoch_losses[3]:.4f}")
        best_model_path =temp_dir/f"{self.hparams.model_path.replace('/', '_')}_epoch{self.hparams.epoch}_lr{self.hparams.lr}_asl{self.hparams.asl}_psl{self.hparams.psl}_hidden_dim{self.hparams.hidden_dim}_noise{self.hparams.noise}_mean_value{self.hparams.mean_value}.pth"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), best_model_path)
       

    def getSpineEmbeddings(self, batch_size):
        ret = []
        self.data_handler.resetDataOrder()
        num_batches = self.data_handler.getNumberOfBatches(batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size,0, False)
                batch_x = torch.from_numpy(batch_x).to(self.dtype).to(self.device)
                batch_y = torch.from_numpy(batch_y).to(self.dtype).to(self.device)
                _, h, _, _ = self.model(batch_x, batch_y)
                ret.extend(h.cpu().numpy())
        return np.array(ret)

    def return_result(self,tokens, embeddings,topk,all_embeddings,inv_vocab,hparams):
        final_result = {}
        self.model.eval()
        with torch.no_grad():
            for token, embedding in zip(tokens, embeddings):
                batch_x = embedding.clone().detach().to(self.device).to(self.dtype)
                batch_y = batch_x.clone().detach().to(self.device)
                _,h,_,_ = self.model(batch_x, batch_y)#h = [[1,2,3],[1,2,3]]
                top_values = torch.topk(h, k=topk, dim=-1)[0].tolist()
                top_idxs = torch.topk(h, k=topk, dim=-1)[1].tolist()
                for small_token, values,top_idx in zip(token, top_values,top_idxs):
                    c_lists = []
                    if small_token not in ["[CLS]", "[SEP]"]:
                        for value,idx in zip(values,top_idx):
                            current_list = []
                            current_list.append(idx)
                            current_list.append(value)
                            c_lists.append(current_list)
                            final_result[small_token] =c_lists
            _, spine_matrix, _, _ = self.model(all_embeddings, all_embeddings)
            top_values, top_indices = torch.topk(spine_matrix, k=hparams.topk_tokens, dim=-1)
            top_indices = top_indices.cpu().numpy()
            top_words = [[inv_vocab[idx] for idx in idxs] for idxs in top_indices]
           
            final_list = []
            tok_topwords = []
            for k,v in final_result.items():
                for item in v:
                    current = [item]
                    tok_topwords.extend(current)
                    tok_topwords.append(top_words[item[0]])
                final_result[k] = tok_topwords
                tok_topwords = []
        return final_result
                       

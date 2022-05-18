import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 512
input_num_symbol = 12
batch_size = 128
output_size = 2
vec_size = input_size * input_num_symbol
type_position = input_size
sig_shape = [-1, input_num_symbol, input_size]

weight_decay = 1e-6
learning_rate = 0.1
save_interval = 100
max_epoch = 800

embedded_dim = input_size
num_heads = 16

##-----activation-----
coder_act = 'tanh'
koop_act = 'relu'
param_act = 'relu'

##-----file direction--
training_previous_dir = './Data/training_previous.csv'
training_present_dir = './Data/training_present.csv'
training_next_dir = './Data/training_next.csv'
training_param_dir = './Data/training_param.csv'
model_dir = './Model'
Transformer_dir = model_dir + '/Koopman_whole'
Transformer_dir_weight_save_dir = Transformer_dir + '/weight_mtx'


##-----encoder---------
num_encoder_block = 16
encoder_drop_rate = 0.1
encoder_dense_dim = 2048
encoder_norm_mode = 'batch'     #'batch' or 'layer'
##-----Intrincic param-
param_num = 3
param_net_width = [embedded_dim, 512, 256, 128, 64, param_num]
beta = 0.01
##-----decoder---------
num_decoder_block = 16
decoder_drop_rate = 0.1
decoder_dense_dim = 2048
decoder_norm_mode = 'batch'     #'batch' or 'layer'
##-----Koopman---------
Koopman_width = [embedded_dim, 256, 256, 256, 256, 256, 256, embedded_dim]

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 256
input_num_symbol = 32
batch_size = 128
output_size = 1
vec_size = input_size * input_num_symbol
type_position = vec_size
num_classes = 1
sig_shape = [-1, input_num_symbol, input_size]

weight_decay = 0
learning_rate = 0.1
save_interval = 100
max_epoch = 2000

embedded_dim = input_size
num_heads = 16

##-----activation-----
coder_act = 'relu'
MLP_act = 'sigmoid'

##-----file direction--
training_data_path = './Data/training.csv'
testing_data_path = './Data/testing.csv'
validation_data_path = './Data/validation.csv'
model_dir = './Model'
Transformer_dir = model_dir + '/Transformer_whole'
Transformer_weight_save_dir = Transformer_dir + '/weight_mtx'


##-----encoder---------
num_encoder_block = 16
encoder_drop_rate = 0.1
encoder_dense_dim = 2048
encoder_norm_mode = 'layer'     #'batch' or 'layer'
##-----decoder---------
num_decoder_block = 0
decoder_drop_rate = 0.1
decoder_dense_dim = 2048
decoder_norm_mode = 'layer'     #'batch' or 'layer'
##-----Koopman---------
Koopman_width = [embedded_dim, 256, 256, 256, 256, 256, 256, embedded_dim]

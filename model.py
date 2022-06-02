import time
import config
from torch import nn
import os
import torch
import network
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict
from torchsummary import summary
import gym


class Transformer_model(nn.Module):
    def __init__(self):
        super(Transformer_model, self).__init__()
        self.Transformer_autoencoder = network.Transformer_net(num_encoder_block=config.num_encoder_block,
                                                               num_decoder_block=config.num_decoder_block,
                                                               act_mode=config.MLP_act).to(
            device=config.device)
        self.criterion = nn.BCELoss().to(device=config.device)
        self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)

    def save(self, epoch):
        checkpoint_path = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % (epoch))
        if not os.path.exists(config.Transformer_dir):
            os.makedirs(config.Transformer_dir, exist_ok=True)
        torch.save(self.Transformer_autoencoder.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location=config.device)
        self.Transformer_autoencoder.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def train(self, encoder_inputs, decoder_inputs, y_train, y_label):
        self.Transformer_autoencoder.train()
        # summary(self.Transformer_autoencoder, [(32, 256), (32, 256)])

        num_samples = encoder_inputs.shape[0]
        num_batches = num_samples // config.batch_size

        print('### Training... ###')
        for epoch in range(1, config.max_epoch + 1):
            shuffle_index = np.random.permutation(num_samples)
            start_time = time.time()
            curr_encoder_inputs = encoder_inputs[shuffle_index]
            curr_decoder_inputs = decoder_inputs[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            curr_y_label = y_label[shuffle_index]
            loss_ = 0

            if epoch == 1:
                config.learning_rate = 0.01
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate,
                                                  weight_decay=config.weight_decay)
            elif epoch == int(config.max_epoch * 0.5):
                config.learning_rate = 0.005
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate,
                                                  weight_decay=config.weight_decay)
            elif epoch == int(config.max_epoch * 0.8):
                config.learning_rate = 0.001
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate,
                                                  weight_decay=config.weight_decay)

            for i in range(num_batches):
                current_batch_encoder_input = torch.tensor(curr_encoder_inputs[config.batch_size * i:
                                                                               config.batch_size * (i + 1), :, :],
                                                           dtype=torch.float, device=config.device)
                current_batch_decoder_input = torch.tensor(curr_decoder_inputs[config.batch_size * i:
                                                                               config.batch_size * (i + 1), :, :],
                                                           dtype=torch.float, device=config.device)
                current_batch_y_train = torch.tensor(np.reshape(curr_y_label[config.batch_size * i:
                                                                             config.batch_size * (i + 1)], (-1, 1)),
                                                     dtype=torch.float, device=config.device)

                curr_batch_y_pred = self.Transformer_autoencoder(current_batch_encoder_input,
                                                                 current_batch_decoder_input)
                loss = self.criterion(curr_batch_y_pred, current_batch_y_train)
                loss_ += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            test = torch.eq(torch.round(self.Transformer_autoencoder(torch.tensor(curr_encoder_inputs[:500, :, :],
                                                                                  dtype=torch.float,
                                                                                  device=config.device),
                                                                     torch.tensor(curr_decoder_inputs[:500, :, :],
                                                                                  dtype=torch.float,
                                                                                  device=config.device))),
                            torch.tensor(np.reshape(curr_y_label[:500], (-1, 1)), device=config.device))
            acc = torch.div(torch.sum(test), 500)
            print('Epoch {:d} Loss {:.6f} Accuracy {:.6f} Duration {:.3f} seconds.'.format(epoch, loss_ / num_batches,
                                                                                           acc,
                                                                                           duration))
            if epoch % config.save_interval == 0:
                self.save(epoch)

    def test_or_validate(self, encoder_inputs, decoder_inputs, y, checkpoint_num_list):
        self.Transformer_autoencoder.eval()
        print('### Test or Validation ###')

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)

            preds = []
            for i in tqdm(range(encoder_inputs.shape[0])):
                encoder_inter = [encoder_inputs[i]]
                decoder_inter = [decoder_inputs[i]]
                out = self.Transformer_autoencoder(torch.tensor(encoder_inter, dtype=torch.float, device=config.device),
                                                   torch.tensor(decoder_inter, dtype=torch.float, device=config.device))
                out = torch.argmax(out, dim=-1)
                preds.append(out.detach().cpu().numpy())

            preds = np.reshape(preds, (-1, config.output_size))
            sum = 0
            for i in range(y.shape[0]):
                if preds[i] == (y[i]):
                    sum += 1
            accuracy = sum / y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy))

    def save_matrix(self, checkpoint_num_list):
        self.Transformer_autoencoder.eval()
        print('### Save Weight Matrices ###')

        isExist = os.path.exists(config.Transformer_weight_save_dir)
        if not isExist:
            os.makedirs(config.Transformer_weight_save_dir)
            print("The new directory is created")

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)
            for params in self.Transformer_autoencoder.state_dict():
                weight = self.Transformer_autoencoder.state_dict()[params].cpu().numpy()
                checkpoint_file = os.path.join(config.Transformer_weight_save_dir,
                                               'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                pd.DataFrame(weight).to_csv(checkpoint_file, header=False, index=False)

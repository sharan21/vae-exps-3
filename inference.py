import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from models.LSTM2 import LSTMClassifier2



def main():

    load_checkpoint = "./bin/2021-Apr-21-16-59-38-lstm2/E0.pytorch"

    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300

    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

    model = LSTMClassifier2(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

    if not os.path.exists(load_checkpoint):
        raise FileNotFoundError(load_checkpoint)

    model.load_state_dict(torch.load(load_checkpoint))
    print("Model loaded from %s" % load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    exit()

    samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':

    main()

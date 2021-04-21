# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils import to_var

class LSTMClassifier2(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(LSTMClassifier2, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		
		self.latent_size = 40
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		# self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)

		 # hidden to style space
		self.hidden2stylemean = nn.Linear(self.hidden_size, int(self.latent_size/4))
		self.hidden2stylelogv = nn.Linear(self.hidden_size, int(self.latent_size/4))

		# hidden to content space
		self.hidden2contentmean = nn.Linear(self.hidden_size, int(3*self.latent_size/4))
		self.hidden2contentlogv = nn.Linear(self.hidden_size, int(3*self.latent_size/4))

		# classifiers
		# self.content_classifier = nn.Linear(int(3*self.latent_size/4), self.content_bow_dim)
		# self.style_classifier_1 = nn.Linear(int(latent_size/4), 10) # for correlating style space to sentiment
		# self.style_classifier_2 = nn.Linear(10, 2) # for correlating style space to sentiment

		# self.style_sigmoid = nn.ReLU()

		# dsicrimimnator/adversaries

		# latent to initial hs for decoder
		self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)

		self.lstm2 = nn.LSTM(embedding_length, hidden_size)

		# final hidden to output vocab
		self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)

		self.embedding_dropout = nn.Dropout(p=0.5)
		
	def forward(self, input_sentence, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''

		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())


		output, (hidden, final_cell_state) = self.lstm(input, (h_0, c_0))
		style_preds = self.label(hidden[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)


		#style component
		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		style_z = to_var(torch.randn([self.batch_size, int(self.latent_size/4)])) #get a random vector
		style_z = style_z * style_std + style_mean #compute datapoint

		#content component

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		content_z = to_var(torch.randn([self.batch_size, int(3*self.latent_size/4)])) #get a random vector
		content_z = content_z * content_std + content_mean #compute datapoint

		 #concat style and concat
	
		final_mean = torch.cat((style_mean[0], content_mean[0]), dim=1)
		final_logv = torch.cat((style_logv[0], content_logv[0]), dim=1)
		final_z = torch.cat((style_z[0], content_z[0]), dim=1)

		hidden2 = self.latent2hidden(final_z)

		# decoder input
		#
		input_embedding = self.embedding_dropout(input)
		# packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

		# decoder forward pass
	
		hidden2 = torch.unsqueeze(hidden2, dim=0)
		outputs, _ = self.lstm2(input_embedding, (hidden2, c_0))
		
		outputs = outputs.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

		
		final_tokens = self.outputs2vocab(outputs)
		# print(final_tokens[0, 0:10, 0:10])
		
		final_tokens = nn.functional.log_softmax(final_tokens, dim = 2)
		final_tokens = torch.exp(final_tokens)
		# print(final_tokens[0, 0, :].sum())
		# exit()


		return final_tokens, final_mean, final_logv, final_z, style_preds

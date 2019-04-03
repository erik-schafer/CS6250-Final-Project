import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MyMLP(nn.Module):
	def __init__(self, n_input=178,n_hidden=16,n_output=5):
		super(MyMLP, self).__init__()
		self.hidden1 = nn.Linear(n_input,n_hidden)
		self.hidden2 = nn.Linear(n_hidden,n_output)

	def forward(self, x):
		x = F.sigmoid(self.hidden1(x))
		x = F.sigmoid(self.hidden2(x))
		#x = self.out(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features=32, out_features=5)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x[:, -1, :])
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.dim_input = dim_input
		self.FC32 = nn.Linear(dim_input, 64)
		self.GRU16 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True, dropout=0.5)
		self.FC2 = nn.Linear(in_features=32, out_features=2)

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		s, l = input_tuple
		dense = F.tanh(self.FC32(s))
		s_p = pack_padded_sequence(dense, l, batch_first=True)
		s, h = self.GRU16(s_p)
		s_out = self.FC2(h[-1,:,:])
		return s_out
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# -: Read a csv file from path.
	# -: Please refer to the header of the file to locate X and y.
	# -: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# -: Remove the header of CSV file of course.
	# -: Do Not change the order of rows.
	# -: You can use Pandas if you want to.
	rawData = pd.read_csv(path)
	#print(rawData.head)
	#rawData['y'] = rawData['y'] - 1
	#print(rawData.head)
	y = (rawData['y'] - 1).to_numpy()#.astype("float32")
	X = rawData.drop(columns="y").to_numpy().astype("float32")
	
	##
	# "Since each model requires a different shape of input data, 
	# you should convert the raw data accordingly. Please look at 
	# the code template, and you can complete each type of conversion 
	# at each stage as you progress"
	# TODO: Foreach Model Transform data as needed...?
	##

	if model_type == 'MLP':
		#data = torch.zeros((2, 2))
		#target = torch.zeros(2)
		scaler = MaxAbsScaler()
		scaler.fit(X)
		X_t = scaler.transform(X)
		data = torch.from_numpy(X_t)
		target = torch.from_numpy(y)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		#data = torch.zeros((2, 2))
		#target = torch.zeros(2)
		#X_t = torch.unsqueeze(X, 1)#X.view(-1, 16*41)#X[:, :]
		data = torch.from_numpy(X)
		data = torch.unsqueeze(data, 1)
		target = torch.from_numpy(y)
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		#data = torch.zeros((2, 2))
		#target = torch.zeros(2)
		data = torch.from_numpy(X)
		data = torch.unsqueeze(data, 2)
		target = torch.from_numpy(y)
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")
	#print(X.size)
	#print(X.shape)
	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# -: Calculate the number of features (diagnoses codes in the train set)
	t = reduce(lambda x,y: x+y, seqs)
	t = reduce(lambda x,y: x+y, t)
	#print(np.unique(np.array(t)))
	return np.unique(np.array(t)).size




class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels
		# -: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# -: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# -: You can use Sparse matrix type for memory efficiency if you want.
		#self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.
		t_seqs = []
		j = 0
		for patient in seqs:
			visitCnt = len(patient)
			p_data = np.zeros((visitCnt, num_features))
			for i, visit in enumerate(patient):
				#print(f"patient: {j}, visit: {i}, icd9s: {visit}")
				p_data[i, np.array(visit)] = 1
			t_seqs += [p_data]
			j += 1
		self.seqs = t_seqs
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# -: Return the following two things
	# -: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# -: 2. Tensor contains the label of each sequence
	batch = sorted(batch, key=lambda x: x[0].shape[0],reverse=True)
	max_length = np.max([s.shape[0] for s,l in batch])
	batch_size = len(batch)
	num_features = np.max([s.shape[1] for s,l in batch])
	#print(batch_size , max_length , num_features)

	labels = []
	lengths = []
	seqs = np.zeros((batch_size, max_length, num_features), dtype='float32')
	i = 0
	for s, l in batch:
		labels += [l]
		length = [s.shape[0]]
		lengths += [s.shape[0]]
		#print(i, length[0])
		seqs[i,0:length[0],:] = s
		i += 1

	#seqs_tensor = torch.FloatTensor()
	#lengths_tensor = torch.LongTensor()
	#labels_tensor = torch.LongTensor()
	seqs_tensor = torch.from_numpy(seqs)#.astype(torch.float32)
	lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
	labels_tensor = torch.tensor(labels, dtype=torch.int64)

	return (seqs_tensor, lengths_tensor), labels_tensor

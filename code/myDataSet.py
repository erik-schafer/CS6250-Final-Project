import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset

class Xray(Dataset):
    def __init__(self, labels_path, image_path, train_path, test_path, train=True,tansform=None, batchSize=28):
        # Store configuration
        self.labels_path = labels_path 
        self.image_path = image_path
        self.batchSize = batchSize
        self.train = train

        # get the list of images in the train image folder
        self.trainingIdxs = []
        with open(train_path, 'r') as f:
            for l in f.readlines():
                self.trainingIdxs += [l.strip()]

        # get the list of images in the test image folder
        self.testingIdxs = []
        with open(test_path, 'r') as f:
            for l in f.readlines():
                self.testingIdxs += [l.strip()]
        
        # call load labels function
        self.trainData, self.testData, self.encoder = self.loadLabels()
        
        # print some output for sanity check
        print("train, test", len(self.trainData), len(self.testData))
    
    def loadLabels(self):
        # read the labels file
        data = pd.read_csv(self.labels_path)

        # Transform and store findings like "Disease 1|Finding 2|Disease 2" to ["Disease 1", "Finding 2", "Disease 2"]
        data["y_list"] = data["Finding Labels"].apply(lambda x: x.split("|"))

        # instansiate a MultiLabelBinarizer to map list of outputs like ["Disease 1", "Finding 2", "Disease 2"] to one-hot vector
        encoder = MultiLabelBinarizer()
        encoder.fit(data["y_list"])        

        # find the "No Finding"
        noFindingIndex = np.where(encoder.classes_ == "No Finding")[0][0]
        # remove the no finding index, since a vector of <0,0,0...,0> will correspond to "No Finding"
        data["y"] = np.delete(encoder.transform(data["y_list"]), [noFindingIndex], axis=1).astype("float").tolist()
        
        # get the image index (filename) and the output vector for train test
        trainData = data[data["Image Index"].isin(self.trainingIdxs)][["Image Index", "y"]]
        testData = data[data["Image Index"].isin(self.testingIdxs)][["Image Index", "y"]]

        # get a list of all images on disk
        images = os.listdir(self.image_path)

        # clip the train and test data to exclude any images not on disk
        trainData = trainData[trainData["Image Index"].isin(images)]
        testData = testData[testData["Image Index"].isin(images)]

        # set index on the df so that they are more useful later
        trainData.set_index("Image Index", inplace=True)
        testData.set_index("Image Index", inplace=True)
        
        # return values
        return trainData, testData, encoder
    
    
    def __len__(self):
        # len override for train and test
        if self.train:
            return len(self.trainData)
        else:
            return len(self.testData)

    def __getitem__(self, idx):
        # set data based on whether this DataSet is for training or testing
        if self.train:
            data = self.trainData
        else:
            data = self.testData
        # convert the output vector to tensor
        y = torch.FloatTensor(data.iloc[idx]['y'])
        
        # get the filename from the data at index
        fn = data.iloc[idx]
        # get the image from disk
        im = io.imread(os.path.join(self.image_path, fn.name))
        # convert the image to tensor
        X = torchvision.transforms.functional.to_tensor(im).float()        
        # return the X,y pair.
        return X, y

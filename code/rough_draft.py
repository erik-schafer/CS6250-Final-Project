import os
import pickle

import pandas as pd
import numpy as np


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from skimage import io, transform
import skimage

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Define Directories
IMAGE_DIRECTORY = os.path.join("../data","images")
LABEL_CSV = os.path.join("../data", "Data_Entry_2017.csv")
BBox_CSV = "BBox_List_2017.csv"

IMAGE_OUT = os.path.join("../data", "image_out")
FEATURES_OUT = os.path.join("../data", "features")

# Data Set object for training
class Xray(Dataset):
    def __init__(self, labels_path, image_path, transform_shape, tansform=None, testing=True):
        self.labels_path = labels_path
        self.image_path = image_path
        self.transform_shape = transform_shape

        #self.encoder
        self.labels = self.loadLabels()
        self.images, self.indices = self.loadImages()
        self.reduceLabels()
        self.labels.set_index("Image Index", inplace=True)
        return
    
    def loadLabels(self):
        data = pd.read_csv(self.labels_path)
        # Transform and store findings like "Disease 1|Finding 2|Disease 2" to ["Disease 1", "Finding 2", "Disease 2"]
        data["y_list"] = data["Finding Labels"].apply(lambda x: x.split("|"))

        encoder = MultiLabelBinarizer()
        encoder.fit(data["y_list"])

        noFindingIndex = np.where(encoder.classes_ == "No Finding")[0][0]

        data["y"] = np.delete(encoder.transform(data["y_list"]), [noFindingIndex], axis=1).astype("float").tolist()
        
        return data[["Image Index", "y"]]

    def loadImages(self):
        lim = -1#10#00
        images = []
        indices = []
        i = 0
        for f in os.listdir(self.image_path):
            f_path = os.path.join(self.image_path, f)
            try:
                im = io.imread(f_path)
                im = transform.resize(im, self.transform_shape)
                im = np.stack([im]*3, axis=-1)
                print(im.shape)
                #im = im.reshape((TRANSFORM_SHAPE[0], TRANSFORM_SHAPE[1], 1)) # from shape (256,256,1)
                images += [im]
                indices += [f]
            except Exception:
                print(f)
            if i > lim: break
            if i % 100 == 0: print(f)
            i += 1
        return images, indices

    def reduceLabels(self):
        self.labels = self.labels[self.labels["Image Index"].isin(self.indices)]

    def loadLabels(self):
        data = pd.read_csv(LABEL_CSV)
        # Transform and store findings like "Disease 1|Finding 2|Disease 2" to ["Disease 1", "Finding 2", "Disease 2"]
        data["y_list"] = data["Finding Labels"].apply(lambda x: x.split("|"))

        encoder = MultiLabelBinarizer()
        encoder.fit(data["y_list"])

        noFindingIndex = np.where(encoder.classes_ == "No Finding")[0][0]

        data["y"] = np.delete(encoder.transform(data["y_list"]), [noFindingIndex], axis=1).tolist()
        #return data[["Image Index", "Patient ID", "y"]]
        return data[["Image Index", "y"]]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imName = self.indices[idx]
        image = self.images[idx]
        label = self.labels.loc[imName]
        X = torchvision.transforms.functional.to_tensor(image).float()
        y = torch.FloatTensor(label["y"])
        return X, y





## Copied
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def train(model, train_loader, criterion, optimizer, num_epochs=4):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X) # batch outputs
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} running_loss {running_loss}")
    return model

def main():
    ds = Xray(LABEL_CSV, IMAGE_DIRECTORY, TRANSFORM_SHAPE)
    model = DenseNet121(14)
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss() # doesn't work multilabel output
    optimizer = optim.Adam(model.parameters())
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=False)
    
    model.train() # set training mode
    




if __name__== "__main__":
    main()
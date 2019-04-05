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

# Set cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Define Directories
IMAGE_DIRECTORY = os.path.join("../data","images")
IMAGE_TRANSFORM_DIRECTORY = os.path.join("../data","image_transform")
LABEL_CSV = os.path.join("../data", "Data_Entry_2017.csv")

TRAIN_PATH = os.path.join("../data", "train_val_list.txt")
TEST_PATH = os.path.join("../data", "test_list.txt")

TRANSFORM_SHAPE = (256,256)

BATCH_SIZE = 28

# Data Set object for training
class Xray(Dataset):
    def __init__(self, labels_path, image_path, train_path, test_path, train=True,tansform=None, batchSize=BATCH_SIZE*10):
        self.labels_path = labels_path
        self.image_path = image_path
        self.batchSize = batchSize
        self.train = train

        self.trainingIdxs = []
        with open(train_path, 'r') as f:
            for l in f.readlines():
                self.trainingIdxs += [l.strip()]

        self.testingIdxs = []
        with open(test_path, 'r') as f:
            for l in f.readlines():
                self.testingIdxs += [l.strip()]
        self.trainData, self.testData, self.encoder = self.loadLabels()
        #self.trainData, self.testData = self.trainData[:50], self.testData[:50]
        print("train, test", len(self.trainData), len(self.testData))
    
    def test(self):
        self.train = False
    
    def train(self):
        self.train = True
    
    def loadLabels(self):
        data = pd.read_csv(self.labels_path)
        # Transform and store findings like "Disease 1|Finding 2|Disease 2" to ["Disease 1", "Finding 2", "Disease 2"]
        data["y_list"] = data["Finding Labels"].apply(lambda x: x.split("|"))

        encoder = MultiLabelBinarizer()
        encoder.fit(data["y_list"])        

        noFindingIndex = np.where(encoder.classes_ == "No Finding")[0][0]

        data["y"] = np.delete(encoder.transform(data["y_list"]), [noFindingIndex], axis=1).astype("float").tolist()
        
        trainData = data[data["Image Index"].isin(self.trainingIdxs)][["Image Index", "y"]]
        testData = data[data["Image Index"].isin(self.testingIdxs)][["Image Index", "y"]]

        images = os.listdir(self.image_path)

        trainData = trainData[trainData["Image Index"].isin(images)]
        testData = testData[testData["Image Index"].isin(images)]

        trainData.set_index("Image Index", inplace=True)
        testData.set_index("Image Index", inplace=True)
        
        return trainData, testData, encoder
    
    
    def __len__(self):
        if self.train:
            return len(self.trainData)
        else:
            return len(self.testData)

    def __getitem__(self, idx):
        if self.train:
            data = self.trainData
        else:
            data = self.testData
        y = torch.FloatTensor(data.iloc[idx]['y'])
        fn = data.iloc[idx]
        im = io.imread(os.path.join(self.image_path, fn.name))
        X = torchvision.transforms.functional.to_tensor(im).float()
        #X = self.imBatch[idx % self.batchSize]
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

def train(model, train_loader, criterion, optimizer, test_loader, num_epochs=4):
    for epoch in range(num_epochs):
        model.train()
        print(f"begin epoch {epoch}")
        running_loss = 0.0
        printMemoryDetails()
        for i, (X, y) in enumerate(train_loader):
            X = X.to(0)
            y = y.to(0)
            optimizer.zero_grad()
            outputs = model(X) # batch outputs
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"\t\t\tepoch {epoch} running_loss {running_loss}")
        print(f"Testing...")
        t, p = validate(model, test_loader)

        with open(f"target-{epoch}.pkl","wb") as f:
            pickle.dump(t, f)
        with open(f"pred-{epoch}.pkl","wb") as f:
            pickle.dump(p, f)
        print(roc_auc_score(t,p))
        torch.save(model.state_dict(), f"model-{epoch}.pkl")
    return model

def validate(model, valid_loader):
    model.eval()
    with torch.no_grad():
        target = torch.FloatTensor()
        pred = torch.FloatTensor()
        for i, (X,y) in enumerate(valid_loader):
            #print(i, end=", ")
            #X = X.to(0)
            #y = y.to(0)
            X = X.view(-1,3,TRANSFORM_SHAPE[0], TRANSFORM_SHAPE[1]).to(0)
            prediction = model(X).cpu()
            target = torch.cat((target, y), 0)
            pred = torch.cat((pred, prediction), 0)
    return target, pred

def printMemoryDetails():
    print(f"curent device {torch.cuda.current_device()}")
    print(f"memory allocated {torch.cuda.memory_allocated()}")
    print(f"memory cached {torch.cuda.memory_cached()}")
    print(f"max memory allocated {torch.cuda.max_memory_allocated()}")

def main():
    
    print("making Dataset... ", end="")
    # labels_path, image_path, train_path, test_path
    ds = Xray(LABEL_CSV, IMAGE_TRANSFORM_DIRECTORY, TRAIN_PATH, TEST_PATH)
    ds_v = Xray(LABEL_CSV, IMAGE_TRANSFORM_DIRECTORY, TRAIN_PATH, TEST_PATH, train=False)

    print(ds.encoder.classes_)
    print(ds_v.encoder.classes_)

    print(f"train, test: ({len(ds)}, {len(ds_v)})")

    print("making model")
    model = DenseNet121(14).cuda()
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss() # doesn't work multilabel output
    optimizer = optim.Adam(model.parameters())
    
    print("dataloaders")
    dl_train = DataLoader(dataset=ds, batch_size=28, shuffle=False, pin_memory=True)
    dl_test = DataLoader(dataset=ds_v , batch_size=28, shuffle=False,  pin_memory=True)    

    print("training")    
    model = train(model, dl_train, criterion, optimizer, dl_test)
    




if __name__== "__main__":
    main()
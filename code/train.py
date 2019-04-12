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

# local imports
from myModel import DenseNet121
from myDataSet import Xray


# Set cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Define Directories
IMAGE_DIRECTORY = os.path.join("../data","images")
IMAGE_TRANSFORM_DIRECTORY = os.path.join("../data","image_transform")
LABEL_CSV = os.path.join("../data", "Data_Entry_2017.csv")

TRAIN_PATH = os.path.join("../data", "train_val_list.txt")
TEST_PATH = os.path.join("../data", "test_list.txt")

TRANSFORM_SHAPE = (256,256)

BATCH_SIZE = 32

def train(model, train_loader, criterion, optimizer, test_loader, num_epochs=4):
    for epoch in range(num_epochs):
        # set the model to "train" state
        model.train()
        # output for sanity
        print(f"begin epoch {epoch}")
        # initialize running_loss for this epoch to be zero
        running_loss = 0.0
        # printMemoryDetails()
        for i, (X, y) in enumerate(train_loader):
            # move tensors to GPU
            X = X.to(0)
            y = y.to(0)
            # reset optimizer
            optimizer.zero_grad()
            # collect outputs
            outputs = model(X) # batch outputs
            # collect loss
            loss = criterion(outputs, y)
            # backprop
            loss.backward()
            # iterate optimizer
            optimizer.step()
            # store running loss
            running_loss += loss.item()
        # outuput
        print(f"\t\t\tepoch {epoch} running_loss {running_loss}")
        print(f"Testing...")
        # use the test loader to get targets, predictions
        t, p = validate(model, test_loader)

        # store targets, predictions for later use (e.g. graphs)
        with open(f"target-{epoch}.pkl","wb") as f:
            pickle.dump(t, f)
        with open(f"pred-{epoch}.pkl","wb") as f:
            pickle.dump(p, f)
        # compute, print ROC AUC score
        print(roc_auc_score(t,p))
        # save the model for later use 
        torch.save(model.state_dict(), f"model-{epoch}.pkl")
    return model

def validate(model, valid_loader):
    # set the model to eval state
    model.eval()
    # no_grad to avoid computing any gradients
    with torch.no_grad():
        # set target, pred as tensor variables
        target = torch.FloatTensor()
        pred = torch.FloatTensor()
        # foreach X,y...
        for i, (X,y) in enumerate(valid_loader):
            # reshape (not sure why?) and move to GPU
            X = X.view(-1,3,TRANSFORM_SHAPE[0], TRANSFORM_SHAPE[1]).to(0)
            # get the prediction from the model, move it to CPU
            prediction = model(X).cpu()
            # concat targets
            target = torch.cat((target, y), 0)
            # concat predictions
            pred = torch.cat((pred, prediction), 0)
    # return targets, predictions        
    return target, pred

def printMemoryDetails():
    print(f"curent device {torch.cuda.current_device()}")
    print(f"memory allocated {torch.cuda.memory_allocated()}")
    print(f"memory cached {torch.cuda.memory_cached()}")
    print(f"max memory allocated {torch.cuda.max_memory_allocated()}")

def main():
    
    print("making Dataset... ", end="")
    # initialize a trainign and testing dataset
    ds = Xray(LABEL_CSV, IMAGE_TRANSFORM_DIRECTORY, TRAIN_PATH, TEST_PATH)
    ds_v = Xray(LABEL_CSV, IMAGE_TRANSFORM_DIRECTORY, TRAIN_PATH, TEST_PATH, train=False)

    # sanity check (and also because I'm not storing the encodings -- should always be the same)
    print(ds.encoder.classes_)
    print(ds_v.encoder.classes_)

    # print size of our datasets, for sanity check
    print(f"train, test: ({len(ds)}, {len(ds_v)})")

    print("making model")
    # instansiate model, move to GPU
    model = DenseNet121(14).cuda()
    # instansiate loss func
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss() # doesn't work multilabel output
    # instansiate optimizer
    optimizer = optim.Adam(model.parameters())
    
    print("dataloaders")
    dl_train = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    dl_test = DataLoader(dataset=ds_v , batch_size=BATCH_SIZE, shuffle=False,  pin_memory=True)    

    print("training")    
    model = train(model, dl_train, criterion, optimizer, dl_test, 30)
    


if __name__== "__main__":
    main()
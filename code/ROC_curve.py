import myModel

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import pickle


# the best epoch was epoch 10
t1 = 'target-10.pkl'
p1 = 'pred-10.pkl'

# m1 = 'model-10.pkl'
# m = myModel.DenseNet121(14)
# m.load_state_dict(torch.load(m1))
# model.eval()

# unpickle the target, prediction
with open(t1, 'rb') as f:
    t1 = pickle.load(f)

with open(p1, 'rb') as f:
    p1 = pickle.load(f)

# false/true positive rate dictionaries
fpr = dict()
tpr = dict()
roc_auc = dict()

# convert to np array
prediction = np.array(p1)
target = np.array(t1)

# for each classification
for i in range(14):
    # compute the fpr, tpr
    fpr[i], tpr[i], _ = roc_curve(target[:, i], prediction[:, i])
    # compute the total auc
    roc_auc[i] = auc(fpr[i], tpr[i])


# generate plots
categories = ['Atelectasis','Cardiomegaly','Consolidation',
            'Edema','Effusion','Emphysema','Fibrosis','Hernia',
            'Infiltration','Mass','Nodule','Pleural_Thickening',
            'Pneumonia','Pneumothorax']
for i, c in enumerate(categories):
    plt.figure()
    plt.plot(fpr[i], tpr[i], color='darkorange',
            label=f'ROC curve (area = {roc_auc[i]})' )
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {c}')
    plt.legend(loc="lower right")
    plt.savefig(f"ROC {c}.png")


ROC_BY_EPOCH = [0.5413986150765521, 0.5839276932023587, 0.6092466162223547, 0.6411124689101696, 0.6694322585247002, 0.6859429647530277, 0.6925890959474744, 0.7020508878893157, 0.7069498442603415, 0.7076733706641151, 0.707437253762425, 0.7057425990601455, 0.6996866960288434, 0.6939237963825062, 0.6844533384249499, 0.6769798784868472, 0.6659911589015793, 0.6661886120033552, 0.6633194117983079, 0.6470170518200167, 0.6378082510260176, 0.6460975644981913, 0.6354241499670769, 0.6378734258328442, 0.6464229913924077]
plt.figure()
plt.plot(ROC_BY_EPOCH)
plt.xlim([0, len(ROC_BY_EPOCH)])
plt.ylim([np.min(ROC_BY_EPOCH), 1])
plt.title(f"Total ROC score by training Epoch")
plt.savefig("ROC BY EPOCH.png")
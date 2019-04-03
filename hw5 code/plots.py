import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# -: Make plots for loss curves and accuracy curves.
	# -: You do not have to return the plots.
	# -: You can save plots as files by codes here or an interactive way according to your preference.
	# TODO: Validate results from this method
	print("plot_learning_curves")
	print(train_losses, valid_losses)
	plt.title("Loss Curve")
	plt.plot(train_losses, label="Training Loss")
	plt.plot(valid_losses, label="Validation Loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.savefig("loss.png")
	plt.cla()
	
	print(train_accuracies, valid_accuracies)
	plt.title("Accuracy Curve")
	plt.plot(train_accuracies, label="Training Accuracy")
	plt.plot(valid_accuracies, label="Validation Accuracy")
	plt.xlabel("epoch")
	plt.ylabel("Accuracy")
	plt.savefig("accu.png")



def plot_confusion_matrix(results, class_names):
	# -: Make a confusion matrix plot.
	# -: You do not have to return the plots.
	# -: You can save plots as files by codes here or an interactive way according to your preference.
	# TODO: Validate this dumpsterfire
    results = np.array(results)
    #print(type(results))
    #print(results)
    np.set_printoptions(precision=2)
    title = "Normalized Confusion Matrix"
    y_true, y_pred = results[:,0], results[:,1]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=class_names, yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("confusion.png")

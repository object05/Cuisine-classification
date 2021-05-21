import numpy as np
import re
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer 

def preprocess(text_list):
    
    """
    This function cleans and pre-processes texts by lemmatizing and removing non-alphabet characters
    """
    
    cleaned=[]
    
    for line in text_list:
        cleaned.append((WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))))

    cleaned = ','.join(cleaned).strip()
    
    return cleaned


def plot_confusion_matrix(cm,
                          target_names,
                          title,
                          cmap=None,
                          normalize=True):
    
    """
    This function compute and plots confusion matrix of a classification model
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold', fontsize=16)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    plt.tight_layout()
    plt.ylabel('True label', weight='bold', fontsize=16)
    plt.xlabel('\n Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), weight='bold', fontsize=16)
    plt.show()
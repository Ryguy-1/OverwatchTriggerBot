from Screen_Data_Gather import ScreenRecord
# python 3.7
# Scikit-learn ver. 0.23.2
from imblearn import pipeline
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
# Imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# OpenCv
import cv2
# matplotlib 3.3.1
from matplotlib import pyplot
# Numpy
import numpy as np
from sklearn.utils import multiclass
# DataLoader
from DataLoader import Dataset
# from DataLoader import MiniBatch
# Pickle
import pickle
# Live Bot
from Live_Bot import Live_Bot


# Hyperparameters
epochs = 5
batch_size = 1584
max_each_class = 990

def train_set():
    data_loader = Dataset('Game_Data_3/', batch_size=batch_size, max_each_class=max_each_class)
    # Verify Sizes
    print(f'Data Loader Length: {len(data_loader)}')
    # Declare Model

    test_x, test_y = data_loader.get_test_data()
    train_x, train_y = data_loader.get_next_batch()
    print(len(train_x))
    print(len(train_y))

    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', verbose=1) # verbose=1
    model.fit(train_x, train_y)

    save_model(model)

    plot_confusion_matrix(model, test_x, test_y)
    pyplot.show()
    test(model, test_x, test_y)


def train():
    
    data_loader = Dataset('Game_Data_2/', batch_size=batch_size, max_each_class=max_each_class)
    # Verify Sizes
    print(f'Data Loader Length: {len(data_loader)}')
    # Declare Model
    model = SGDClassifier(random_state=0, loss='log', penalty='l2') # verbose=1
    # Train Loop
    # Set Rounds Per Epoch
    rounds_per_epoch = int(len(data_loader)/batch_size)
    # Test Data
    test_x, test_y = data_loader.get_test_data()

    for epoch in range(epochs):
        for round in range(rounds_per_epoch):
            data, label = data_loader.get_next_batch()
            model.partial_fit(data, label, classes=data_loader.num_classes_list)
            if round % 3 == 0:
                print(f'Batches Checked: {round}/{rounds_per_epoch}')
        print(f'Epoch: {epoch}/{epochs}')
        test(model, test_x, test_y)
        data_loader.reset_index()
        plot_confusion_matrix(model, test_x, test_y)
    plot_confusion_matrix(model, test_x, test_y)
    pyplot.show()
    
    for i in range(len(test_y)):
        check_image(test_x[i], test_y[i], model)

def test(model, test_x_final, test_y_final):
    preds = model.predict(test_x_final)
    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, test_y_final):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
    pyplot.show()

def check_image(image, label, model):
    # Labels
    label_dict = {
            'clicked': 0,
            'not_clicked': 1,
    }
    # Reverse label_dict
    reverse_label_dict = {v: k for k, v in label_dict.items()}


    print(f'Label: {reverse_label_dict[label]}')
    predictions = model.predict(image.reshape(1, -1))
    print(f'Prediction {reverse_label_dict[predictions[0]]}')
    image = image.reshape(round(1920*3/5-1920*2/5), round(1080*3/5-1080*2/5), 3)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_model(model):
    with open('model_3.pickle', 'wb') as f:
        pickle.dump(model, f)

def gather_data():
    screen_record = ScreenRecord()
    screen_record.start_recording()

if __name__ == '__main__':
    # gather_data()
    # train()
    train_set()
    # live_bot = Live_Bot()
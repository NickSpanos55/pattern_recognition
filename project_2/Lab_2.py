# Νικόλαος Χάιδος 03118096
# Νικόλαος Σπανός 03118822

import os
from glob import glob
import sys
import librosa
from pomegranate import *
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import re
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RFC

import warnings
warnings.filterwarnings("ignore")



# Step 2

def split_fname(fname):
    return re.split('(\d+)',fname)

string_to_int = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [split_fname(f.split("/")[1].split(".")[0].split("\\")[1]) for f in files]
    ids = [f[0]+f[1] for f in fnames]
    
    y = [string_to_int[f[0]] for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)
        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers




# Step 3

def extract_features(wavs, n_mfcc=13, Fs=8000):
    # Extract MFCCs for all wavs
    window = 25 * Fs // 1000
    step = 10 * Fs // 1000
    frames = [
        librosa.feature.mfcc(
            wav, sr=Fs, n_fft=window, hop_length=step, n_mfcc=n_mfcc
        )

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]
    
    # Extract Deltas
    delta_frames = [
        librosa.feature.delta(
            frame, order=1
        )

        for frame in tqdm(frames, desc="Calculating frist-order derivative of features...")
    ]

    # Extract Delta-Deltas
    delta_delta_frames = [
        librosa.feature.delta(
            frame, order=2
        )

        for frame in tqdm(frames, desc="Calculating second-order derivative of features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    frames = [frame.T for frame in frames]
    delta_frames = [delta_frame.T for delta_frame in delta_frames]
    delta_delta_frames = [delta_delta_frame.T for delta_delta_frame in delta_delta_frames]
    
    return frames, delta_frames, delta_delta_frames




# Step 4

wavs, Fs, ids, y, speakers = parse_free_digits("pr_lab2_2020-21_data/digits")
frames, delta_frames, delta_delta_frames = extract_features(wavs, n_mfcc=13, Fs=Fs)

n1 = 2     # 03118822
n2 = 6     # 03118096

mfcc1_n1 = []
mfcc2_n1 = []
mfcc1_n2 = []
mfcc2_n2 = []

# Exctracting 1st and 2nd MFCC for Digits 2 and 6
for idx, frame in enumerate(frames):
    if y[idx]==n1:
        mfcc1_n1.append(frame[:, 0].tolist())
        mfcc2_n1.append(frame[:, 1].tolist())
        
    if y[idx]==n2:
        mfcc1_n2.append(frame[:, 0].tolist())
        mfcc2_n2.append(frame[:, 1].tolist())
        
mfcc1_n1 = sum(mfcc1_n1, [])
mfcc2_n1 = sum(mfcc2_n1, [])
mfcc1_n2 = sum(mfcc1_n2, [])
mfcc2_n2 = sum(mfcc2_n2, [])


fig = plt.figure(figsize=(13,8))

ax1 = fig.add_subplot(2, 2, 1)
plt.hist(mfcc1_n1, bins=40)
ax1.title.set_text('1st MFCC of digit {}'.format(n1))

ax2 = fig.add_subplot(2, 2, 2)
plt.hist(mfcc2_n1, bins=40, color='green')
ax2.title.set_text('2nd MFCC of digit {}'.format(n1))

ax3 = fig.add_subplot(2, 2, 3)
plt.hist(mfcc1_n2, bins=40)
ax3.title.set_text('1st MFCC of digit {}'.format(n2))

ax4 = fig.add_subplot(2, 2, 4)
plt.hist(mfcc2_n2, bins=40, color = 'green')
ax4.title.set_text('2nd MFCC of digit {}'.format(n2))

plt.show()


# wavs[90], speaker 10, digit 6
# wavs[91], speaker 11, digit 6
# wavs[120], speaker 11, digit 2
# wavs[119], speaker 10, digit 2

window = 25 * Fs // 1000      # 25ms Window
step = 10 * Fs // 1000        # 10ms Step

mfsc_n2_10 = librosa.feature.melspectrogram(wavs[90], sr=Fs, n_fft=window, hop_length=step, n_mels=13)
mfsc_n2_11 = librosa.feature.melspectrogram(wavs[91], sr=Fs, n_fft=window, hop_length=step, n_mels=13)
mfsc_n1_10 = librosa.feature.melspectrogram(wavs[119], sr=Fs, n_fft=window, hop_length=step, n_mels=13)
mfsc_n1_11 = librosa.feature.melspectrogram(wavs[120], sr=Fs, n_fft=window, hop_length=step, n_mels=13)


fig = plt.figure(figsize=(13,8))

ax1 = fig.add_subplot(2, 2, 1)
plt.imshow(np.corrcoef(mfsc_n2_10, rowvar=True))
ax1.title.set_text('MFSC Correlation of digit {} from speaker {}'.format(n2, 10))

ax2 = fig.add_subplot(2, 2, 2)
plt.imshow(np.corrcoef(mfsc_n2_11, rowvar=True), cmap = 'inferno')
ax2.title.set_text('MFSC Correlation of digit {} from speaker {}'.format(n2, 11))

ax3 = fig.add_subplot(2, 2, 3)
plt.imshow(np.corrcoef(mfsc_n1_10, rowvar=True))
ax3.title.set_text('MFSC Correlation of digit {} from speaker {}'.format(n1, 10))

ax4 = fig.add_subplot(2, 2, 4)
plt.imshow(np.corrcoef(mfsc_n1_11, rowvar=True), cmap = 'inferno')
ax4.title.set_text('MFSC Correlation of digit {} from speaker {}'.format(n1, 11))

plt.show()



fig = plt.figure(figsize=(13,8))

ax1 = fig.add_subplot(2, 2, 1)
plt.imshow(np.corrcoef(frames[90], rowvar=False))
ax1.title.set_text('MFCC Correlation of digit {} from speaker {}'.format(n2, 10))

ax2 = fig.add_subplot(2, 2, 2)
plt.imshow(np.corrcoef(frames[91], rowvar=False), cmap = 'inferno')
ax2.title.set_text('MFCC Correlation of digit {} from speaker {}'.format(n2, 11))

ax3 = fig.add_subplot(2, 2, 3)
plt.imshow(np.corrcoef(frames[119], rowvar=False))
ax3.title.set_text('MFCC Correlation of digit {} from speaker {}'.format(n1, 10))

ax4 = fig.add_subplot(2, 2, 4)
plt.imshow(np.corrcoef(frames[120], rowvar=False), cmap = 'inferno')
ax4.title.set_text('MFCC Correlation of digit {} from speaker {}'.format(n1, 11))

plt.show()




# Step 5

mean_frames = np.array([np.mean(frame, axis=0) for frame in frames])
var_frames = np.sqrt(np.array([np.var(frame, axis=0) for frame in frames]))
mean_delta_frames = np.array([np.mean(delta_frame, axis=0) for delta_frame in delta_frames])
var_delta_frames = np.sqrt(np.array([np.var(delta_frame, axis=0) for delta_frame in delta_frames]))
mean_delta_delta_frames = np.array([np.mean(delta_delta_frame, axis=0) for delta_delta_frame in delta_delta_frames])
var_delta_delta_frames = np.sqrt(np.array([np.var(delta_delta_frame, axis=0) for delta_delta_frame in delta_delta_frames]))

# Concatenate all the above Features into a single Feature Vector for each Wav File
# In total there will be 13*2*3 = 78 Features
dataset = np.hstack([mean_frames, var_frames, mean_delta_frames, var_delta_frames, mean_delta_delta_frames, var_delta_delta_frames])


markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow', 'pink', 'gray']

for digit in range(1, 10):
    to_plot = np.array(y)==digit
    plt.scatter(dataset[to_plot, 0], dataset[to_plot, 1], c=colors[digit], marker=markers[digit])

plt.title("Scatterplot of Data-Points ")
plt.legend(np.arange(1,10), loc="upper left")
plt.show()




# Step 6

# Lowering Dimensionality with PCA feature-transformation to 2 Dimensions
pca = PCA(n_components=2,svd_solver='full')
dataset_2d = pca.fit_transform(dataset)      # Fit the Transformer, and apply to Train-Data Samples
final_variance_ratios = pca.explained_variance_ratio_

print(f"Total Variance Ratio kept from original Dataset: {final_variance_ratios.sum()*100:.2f}%")

# Different Markers and Colors for each Digit
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow', 'pink', 'gray']

for digit in range(1, 10):
    to_plot = (np.array(y)==digit)
    plt.scatter(dataset_2d[to_plot, 0], dataset_2d[to_plot, 1], c=colors[digit], marker=markers[digit])

plt.title("Scatterplot of Data-Points after Dimensionality Reduction (d=2)")
plt.legend(np.arange(1,10), loc="upper left")
plt.show()


# Lowering Dimensionality with PCA feature-transformation to 3 Dimensions
pca = PCA(n_components=3,svd_solver='full')
dataset_3d = pca.fit_transform(dataset)
final_variance_ratios = pca.explained_variance_ratio_

print(f"Total Variance Ratio kept from original Dataset: {final_variance_ratios.sum()*100:.2f}%")

# Different Markers and Colors for each Digit
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow', 'pink', 'gray']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for digit in range(1, 10):
    to_plot = np.array(y)==digit
    ax.scatter(dataset_3d[to_plot, 0], dataset_3d[to_plot, 1], dataset_3d[to_plot, 2], c=colors[digit], marker=markers[digit])

plt.title("Scatterplot of Data-Points after Dimensionality Reduction (d=3)")
plt.legend(np.arange(1,10), loc="upper left")
plt.show()




# Step 7

# Custom NaiveBayes Class from Lab-1
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.final_means = None
        self.final_vars = None
        self.priors = None
        self.variance_zero_check = None


    def fit(self, X, y):
        self.final_means = np.zeros(shape = (9,X.shape[1]))
        self.final_vars = np.zeros(shape = (9,X.shape[1]))

        for i in range(1,10):
            temp = X[y == i,:]
            
            self.final_means[i-1] = np.mean(temp, axis = 0)
            self.final_vars[i-1] = np.var(temp, axis = 0)
        
        if self.use_unit_variance:
            self.final_vars = np.ones(self.final_vars.shape)
        self.priors = np.unique(y.astype(int), return_counts=True)[1] / y.shape[0]
        
        return self


    def predict(self, X):
        posterior = np.empty((X.shape[0],9))

        for digit in range(1,10):
            posterior[:, digit-1] = scipy.stats.multivariate_normal(self.final_means[digit-1,:], self.final_vars[digit-1,:], allow_singular=True).pdf(X[:, :]) * self.priors[digit-1]

        preds = np.argmax(posterior, axis=1) + 1    # Add one to predictions, because np.argmax is zero-indexed, and digits in dataset satrt from 1
        
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        acc = (self.predict(X) == y).sum() / (y.shape[0])
        
        return acc



# Use initial DataSet, to keep maximum amount of information
X_train, X_test, y_train, y_test = train_test_split(dataset, np.array(y), test_size=0.30)

scaler = StandardScaler()                              
X_train = scaler.fit_transform(X_train, y_train)     # Fit Transformer and Apply Standard Scaling to Train Data
X_test = scaler.transform(X_test)                    # Apply Standard Scaling to Test Data (without re-fitting)

# Classifiers Comparison

nbc_clf = CustomNBClassifier(use_unit_variance=False)
nbc_clf.fit(X_train, y_train)

nbsk_clf = GaussianNB(priors = nbc_clf.priors)
nbsk_clf.fit(X_train, y_train)

svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)

mlp_clf = MLP((10, 20, 10), activation='relu', solver='adam')
mlp_clf.fit(X_train, y_train)

rfc_clf = RFC(n_estimators=250)
rfc_clf.fit(X_train, y_train)

print(f"NaiveBayes Custom Classifier achieved Accuracy of {nbc_clf.score(X_test, y_test)*100:.2f}%")
print(f"NaiveBayes SKLearn Classifier achieved Accuracy of {nbsk_clf.score(X_test, y_test)*100:.2f}%")
print(f"SVM SKLearn Classifier achieved Accuracy of {svm_clf.score(X_test, y_test)*100:.2f}%")
print(f"MLP SKLearn Classifier achieved Accuracy of {mlp_clf.score(X_test, y_test)*100:.2f}%")
print(f"RandomForest SKLearn Classifier achieved Accuracy of {rfc_clf.score(X_test, y_test)*100:.2f}%")




# Step 8

# Custom Dataset to make Sinusoidal Signals, to train RNN
class CustomSignalDataset(Dataset):
    def __init__(self, set_size=1000, Fs = 1600):
        self.train_data = []
        
        for _ in range(set_size):
            x = np.random.uniform(0.0, 1/40)      # Random point in one Period of the Sinusoidal Signal
            x_points = torch.linspace(x, x+(10/(Fs)), 10, dtype=torch.float32, requires_grad=False)  # Draw 10 Samples with Sampling-Frequency Fs
            
            sin_points = torch.unsqueeze(torch.sin(2*np.pi*40*x_points), 1)      # Point Sequence for Train Sample
            cos_points = torch.unsqueeze(torch.cos(2*np.pi*40*x_points), 1)      # Point Sequence for Train Label

            self.train_data.append(( sin_points, cos_points ))
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx][0], self.train_data[idx][1]
    
train_data = CustomSignalDataset(1000, 1600)  # Fs=1600 for more discernible plots

trainset = DataLoader(train_data, batch_size=16)


class CustomRNN(torch.nn.Module):

    def __init__(self, rnn_unit="RNN", n_layers=1, hidden_size=10, dropout=0.0):
        '''
                rnn_unit: "RNN", "LSTM" or "GRU"
        '''
        super(CustomRNN, self).__init__()
        if rnn_unit=="RNN":
            self.rnn = torch.nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=n_layers, bidirectional=False, dropout=dropout, batch_first=True) # Vanilla RNN Unit
        elif rnn_unit=="LSTM":
            self.rnn = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers, bidirectional=False, dropout=dropout, batch_first=True) # LSTM Unit
        else:
            self.rnn = torch.nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=n_layers, bidirectional=False, dropout=dropout, batch_first=True) # GRU Unit
        
        self.dense1 = torch.nn.Linear(hidden_size, 1)

        # One RNN Unit (Vanilla RNN, LSTM or GRU), and one Fully-Connected Layer above
        
    def forward(self, signal):
        outputs, _ = self.rnn(signal)
        outputs = self.dense1(outputs)
        return outputs
    
    def predict(self, signal):
        self.eval()      # Disables Dropout Layers inside the recurrent unit
        outputs, _ = self.rnn(torch.unsqueeze(signal, 0))
        outputs = self.dense1(outputs)
        return torch.squeeze(outputs)
    
    def fit(self, EPOCHS=30):
        self.train()     # Enables Dropout Layers inside the recurrent unit
        criterion = torch.nn.MSELoss()        # Mean-Square-Error Loss Function, εφόσον έχουμε Regression Task
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)     # AdamW Optimizer
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)     # Learning_rate Exponential Scheduler
        
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            
            for i, data in enumerate(trainset):
                sin, cos = data
                
                self.zero_grad()
                predictions = self.forward(sin)
                
                loss = criterion(predictions, cos)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_loss = epoch_loss / len(trainset)
                
            print('EPOCH {}, Train Loss {:.3f}'.format(epoch+1, epoch_loss))

#Training RNN with LSTM unit
model = CustomRNN(rnn_unit="LSTM", n_layers=2, hidden_size=20, dropout=0.0)

model.fit(200)

sample = 204
plt.plot(train_data[sample][0].detach().numpy(), 'g') # input
plt.plot(train_data[sample][1].detach().numpy(), 'r') # input
plt.plot(model.predict(train_data[sample][0]).detach().numpy(), 'b') # predictions
plt.title('RNN with LSTM unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()


Fs = 1600     # Same Sampling-Frequency as the one we trained the model with
x_points = torch.linspace(0, 1/10, int(Fs/10), dtype=torch.float32, requires_grad=False)
sin_points = torch.unsqueeze(torch.sin(2*np.pi*40*x_points), 1)
cos_points = torch.unsqueeze(torch.cos(2*np.pi*40*x_points), 1)

plt.plot(x_points, sin_points, 'g')
plt.plot(x_points, cos_points, 'r')
plt.plot(x_points, model.predict(sin_points).detach().numpy(), 'b')
plt.title('RNN with LSTM unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()

#Training RNN with Vanilla RNN unit
model = CustomRNN(rnn_unit="RNN", n_layers=2, hidden_size=20, dropout=0.0)

model.fit(200)

sample = 204
plt.plot(train_data[sample][0].detach().numpy(), 'g') # input
plt.plot(train_data[sample][1].detach().numpy(), 'r') # input
plt.plot(model.predict(train_data[sample][0]).detach().numpy(), 'b') # predictions
plt.title('RNN with Vanilla RNN unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()


Fs = 1600     # Same Sampling-Frequency as the one we trained the model with
x_points = torch.linspace(0, 1/10, int(Fs/10), dtype=torch.float32, requires_grad=False)
sin_points = torch.unsqueeze(torch.sin(2*np.pi*40*x_points), 1)
cos_points = torch.unsqueeze(torch.cos(2*np.pi*40*x_points), 1)

plt.plot(x_points, sin_points, 'g')
plt.plot(x_points, cos_points, 'r')
plt.plot(x_points, model.predict(sin_points).detach().numpy(), 'b')
plt.title('RNN with Vanilla RNN unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()

#Training RNN with GRU unit
model = CustomRNN(rnn_unit="GRU", n_layers=2, hidden_size=20, dropout=0.0)

model.fit(200)

sample = 204
plt.plot(train_data[sample][0].detach().numpy(), 'g') # input
plt.plot(train_data[sample][1].detach().numpy(), 'r') # input
plt.plot(model.predict(train_data[sample][0]).detach().numpy(), 'b') # predictions
plt.title('RNN with GRU unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()


Fs = 1600     # Same Sampling-Frequency as the one we trained the model with
x_points = torch.linspace(0, 1/10, int(Fs/10), dtype=torch.float32, requires_grad=False)
sin_points = torch.unsqueeze(torch.sin(2*np.pi*40*x_points), 1)
cos_points = torch.unsqueeze(torch.cos(2*np.pi*40*x_points), 1)

plt.plot(x_points, sin_points, 'g')
plt.plot(x_points, cos_points, 'r')
plt.plot(x_points, model.predict(sin_points).detach().numpy(), 'b')
plt.title('RNN with GRU unit')
plt.legend(["Input", "Truth", "Prediction"], loc="upper right")
plt.show()

print("\n\n\n")


# Step 9

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test



X_train, X_test, y_train, y_test, spk_train, spk_test = parser('./free-spoken-digit-dataset-master/recordings')

# Stratified Split, so that each Class has equal number of Samples
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_dev = scale_fn(X_dev)
X_test = scale_fn(X_test)



# Step 10

def custom_gmm_hmm(data, n_states, num_mixtures, gmm):
    
    X = np.vstack(data).astype(np.float)
    
    dists = [] # list of probability distributions for the HMM states
    for i in range(n_states):
        if gmm:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, num_mixtures, X)
        else:
            a = MultivariateGaussianDistribution.from_samples(X)
        dists.append(a)

    # For Transition Matrix, we initialize values to 0.5 in all elements with indices [i,i] or [i,i+1], 
    # and we set the final state transition probability to itself equal to 1.0
    trans_mat = ( 0.5*np.diag(np.ones(n_states-1),1)+0.5*np.diag(np.ones(n_states),0) ).astype(np.float) # your transition matrix
    trans_mat[n_states-1,n_states-1] = 1.0

    # Always start from first state
    starts = np.zeros(n_states).astype(np.float)
    starts[0] = 1.0 # your starting probability matrix
    
    # Always end at last state
    ends = np.zeros(n_states).astype(np.float) # your ending probability matrix
    ends[-1] = 1.0

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])
    
    return model



# Step 11

def train(X_train, y_train, n_states, n_mixtures, max_iter, gmm):
    
    # Split the Dataset to sub-sets for each Digit, to train each model seperately
    X_train_sorted = [[] for _ in range(10)]
    
    for sample, label in zip(X_train,y_train):
        (X_train_sorted[label]).append(sample)
    
    # Initialize the list of the 10 models
    model_list = [[] for _ in range(10)]
    
    for i in range(10):   # One model for each digit
        X = X_train_sorted[i]
        model = custom_gmm_hmm(X, n_states, n_mixtures, gmm)
        
        _ = model.fit(X, max_iterations=max_iter)
        
        model_list[i] = model
    
    return model_list



# Step 12

def find_score(model_list, X_val, y_val):
    
    logp_list = np.zeros((len(X_val), 10))
    
    for i in range(len(X_val)):
        for j in range(len(model_list)):
            logp, _ = model_list[j].viterbi(X_val[i])
            logp_list[i, j] = logp
    
    
    preds = np.argmax(logp_list, axis = 1)    # For each Sample, find the model that gave the highest Probability
    score = np.sum(y_val == preds)/(len(y_val))    # Accuracy Score

    return score, preds


# Grid-Search for HyperParameter Optimization
best_accuracy = 0.0
best_states = 0
best_mixtures = 0
best_iters = 0

for iters in range(2, 63, 3):
    for n_states in range(1,5):
        for n_mixtures in range(1,6):
            if n_mixtures>1:
                current_model = train(X_train, y_train, n_states, n_mixtures, iters, True)
            else:
                current_model = train(X_train, y_train, n_states, n_mixtures, iters, False)
            current_acc, _ = find_score(current_model, X_dev, y_dev)
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_states = n_states
                best_mixtures = n_mixtures
                best_iters = iters

print(f"\n\n\nBest Model:\n", f"\tBest States: {best_states}\n", f"\tBest Mixtures: {best_mixtures}\n", f"\tBest Iterations: {best_iters}\n", f"\tBest Accuracy achieved: {best_accuracy}")

# Best Hyperparameters
if best_mixtures > 1:
    model_list = train(X_train, y_train, best_states, best_mixtures, best_mixtures, True)
else:
    model_list = train(X_train, y_train, best_states, best_mixtures, best_mixtures, False)


# Step 13

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

print("\n\n\n")

# Confusion Matrix and Accuracy Score for Validation Set
score, preds = find_score(model_list, X_dev, y_dev)
plot_confusion_matrix(confusion_matrix(y_dev,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Confusion Matrix on Validation Set',
                      cmap=plt.cm.Blues)
print(f"Model Accuracy on Validation Set is {score}\n")

# Confusion Matrix and Accuracy Score for Test Set
score, preds = find_score(model_list, X_test, y_test)
plot_confusion_matrix(confusion_matrix(y_test,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Confusion Matrix on Test Set',
                      cmap=plt.cm.Reds)
print(f"Model Accuracy on Test Set is {score}")


print("\n\n\n")


# Step 14

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  [seq.shape[0] for seq in feats]

        # Sort Labels, Features, and Lengths, to use pack_padded_sequence later

        temp_lengths = sorted(self.lengths, reverse=True)

        self.feats = [x for _, x in sorted(zip(self.lengths, feats), key=lambda pair: pair[0], reverse=True)]  # Sort descending by Lengths
        self.feats = self.zero_pad_and_stack(feats)
        self.feats = torch.from_numpy(self.feats)

        self.labels = [y for _, y in sorted(zip(self.lengths, labels), key=lambda pair: pair[0], reverse=True)]  # Sort descending by Lengths

        self.lengths = torch.Tensor(sorted(self.lengths, reverse=True))
            
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')
            
        self.labels = torch.from_numpy(self.labels).to(torch.long)
                
    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #

        max_length = max([seq.shape[0] for seq in x])    # Find Max Length
        padded = [(np.pad(sample, pad_width=((0, max_length-sample.shape[0]),(0,0)), constant_values=0)) for sample in x]  # Pad only Dimension-1 until Max_length
        padded = np.stack(padded)
        
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


train_dataset = FrameLevelDataset(X_train, y_train)
test_dataset = FrameLevelDataset(X_test, y_test)
validation_dataset = FrameLevelDataset(X_dev, y_dev)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim=6, rnn_size=20, output_dim=10, num_layers=2, bidirectional=False, dropout=0.2, use_padded_seq=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.use_padded_seq = use_padded_seq
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        self.rnn = torch.nn.LSTM(input_size=input_dim, hidden_size=rnn_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dense1 = torch.nn.Linear(self.feature_size, self.feature_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dense2 = torch.nn.Linear(self.feature_size, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        outputs, _ = self.rnn(x)
        if self.use_padded_seq:    # Use transformation, if Pack_Padded_Sequence is used
            outputs, temp_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.0)
        outputs = self.last_timestep(outputs, lengths, self.bidirectional)
        outputs = self.dropout1(outputs)
        outputs = self.dense1(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.dense2(outputs)
        outputs = self.softmax(outputs)

        return outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1).to(torch.int64)
        return outputs.gather(1, idx).squeeze()
    
    def predict(self, x, lengths):     # Same as Forward, but Dropout is disabled for inference
        self.eval()
        outputs, _ = self.rnn(x)
        if self.use_padded_seq:
            outputs, temp_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.0)
        outputs = self.last_timestep(outputs, lengths, self.bidirectional)
        outputs = self.dropout1(outputs)
        outputs = self.dense1(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.dense2(outputs)
        outputs = self.softmax(outputs)

        return outputs
    
    def score(self, test_set):      # Returns total accuracy score, and Predictions Vector for the Confusion Matrix
        self.eval()
        if self.use_padded_seq:
            packed_test_feats = nn.utils.rnn.pack_padded_sequence(test_set.feats, test_set.lengths, batch_first=True, enforce_sorted=True)
            predictions = self.predict(packed_test_feats, test_set.lengths)
        else:
            predictions = self.predict(test_set.feats, test_set.lengths)
        accuracy = torch.sum(torch.argmax(predictions, dim=1) == test_set.labels) / predictions.shape[0]
        return float(accuracy), torch.argmax(predictions, dim=1)
    
    def fit(self, EPOCHS, train_set, val_set, early_stopping=True, patience=10, l2_decay=1e-3):
        self.train()
        criterion = torch.nn.CrossEntropyLoss()        # Cross-Entropy Loss Function, for Classification Task
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, weight_decay=l2_decay)     # AdamW Optimizer
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)     # Learning_rate Exponential Scheduler
        
        epochs = list(range(1,EPOCHS+1))
        best_loss = 100000.0   # initialization for best validation loss
        counter = 0            # counter for Early_Stopping Patience
        train_loss = []
        val_loss = []
        
        if self.use_padded_seq:
            packed_train_feats = nn.utils.rnn.pack_padded_sequence(train_set.feats, train_set.lengths, batch_first=True, enforce_sorted=True)
            packed_val_feats = nn.utils.rnn.pack_padded_sequence(val_set.feats, val_set.lengths, batch_first=True, enforce_sorted=True)
        
        for epoch in range(EPOCHS):
            self.zero_grad()
            if self.use_padded_seq:
                train_predictions = self.forward(packed_train_feats, train_set.lengths)
            else:
                train_predictions = self.forward(train_set.feats, train_set.lengths)
            epoch_train_loss = criterion(train_predictions, train_set.labels)
            epoch_train_loss.backward()
            optimizer.step()
               
            if self.use_padded_seq:
                val_predictions = self.predict(packed_val_feats, val_set.lengths)
            else:
                val_predictions = self.predict(val_set.feats, val_set.lengths)
            epoch_val_loss = criterion(val_predictions, val_set.labels)
            
            train_loss.append(epoch_train_loss.item())
            val_loss.append(epoch_val_loss.item())
            
            print('EPOCH {}, Train Loss {:.3f}, Validation Loss {:.3f}'.format(epoch+1, epoch_train_loss.item(), epoch_val_loss.item()))
            
            if early_stopping:            # Keep track of Validation Loss, in case of Overfitting
                if epoch_val_loss < best_loss:
                    torch.save(self, "./lstm_checkpoint") # checkpoint for best model so far
                    best_loss = epoch_val_loss
                    counter = 0
                else:
                    counter += 1

                if counter == patience:
                    print(' Early Stop, Validation Loss Increasing ')
                    epochs = list(range(1,epoch+2))
                    break
            
        return epochs, train_loss, val_loss


# Question 4
model_1 = BasicLSTM(input_dim=6, rnn_size=32, output_dim=10, num_layers=2, bidirectional=False, dropout=0.0, use_padded_seq=True)
epochs, train_loss, val_loss = model_1.fit(100, train_dataset, validation_dataset, early_stopping=False, patience=5, l2_decay=0)

score, preds = model_1.score(validation_dataset)
plot_confusion_matrix(confusion_matrix(y_dev,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_1 Confusion Matrix on Validation Set',
                      cmap=plt.cm.Blues)
print(f"Model_1 Accuracy on Validation Set is {score}")

score, preds = model_1.score(test_dataset)
plot_confusion_matrix(confusion_matrix(y_test,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_1 Confusion Matrix on Test Set',
                      cmap=plt.cm.Reds)
print(f"Model_1 Accuracy on Test Set is {score}\n\n")

plt.plot(epochs, train_loss, "green")
plt.plot(epochs, val_loss, "blue")
plt.grid(True)
plt.title("Model_1 Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# Question 5
model_2 = BasicLSTM(input_dim=6, rnn_size=32, output_dim=10, num_layers=2, bidirectional=False, dropout=0.3, use_padded_seq=True)
epochs, train_loss, val_loss = model_2.fit(100, train_dataset, validation_dataset, early_stopping=False, patience=5, l2_decay=1e-3)

score, preds = model_2.score(validation_dataset)
plot_confusion_matrix(confusion_matrix(y_dev,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_2 Confusion Matrix on Validation Set',
                      cmap=plt.cm.Blues)
print(f"Model_2 Accuracy on Validation Set is {score}")

score, preds = model_2.score(test_dataset)
plot_confusion_matrix(confusion_matrix(y_test,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_2 Confusion Matrix on Test Set',
                      cmap=plt.cm.Reds)
print(f"Model_2 Accuracy on Test Set is {score}\n\n")

plt.plot(epochs, train_loss, "green")
plt.plot(epochs, val_loss, "blue")
plt.grid(True)
plt.title("Model_2 Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# Question 6
model_3 = BasicLSTM(input_dim=6, rnn_size=32, output_dim=10, num_layers=2, bidirectional=False, dropout=0.3, use_padded_seq=True)
epochs, train_loss, val_loss = model_3.fit(100, train_dataset, validation_dataset, early_stopping=True, patience=7, l2_decay=1e-3)

score, preds = model_3.score(validation_dataset)
plot_confusion_matrix(confusion_matrix(y_dev,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_3 Confusion Matrix on Validation Set',
                      cmap=plt.cm.Blues)
print(f"Model_3 Accuracy on Validation Set is {score}")

score, preds = model_3.score(test_dataset)
plot_confusion_matrix(confusion_matrix(y_test,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_3 Confusion Matrix on Test Set',
                      cmap=plt.cm.Reds)
print(f"Model_3 Accuracy on Test Set is {score}")

epochs= np.arange(1,77)
plt.plot(epochs, train_loss, "green")
plt.plot(epochs, val_loss, "blue")
plt.grid(True)
plt.title("Model_3 Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# Question 7
model_4 = BasicLSTM(input_dim=6, rnn_size=32, output_dim=10, num_layers=2, bidirectional=True, dropout=0.3, use_padded_seq=True)
epochs, train_loss, val_loss = model_4.fit(100, train_dataset, validation_dataset, early_stopping=True, patience=7, l2_decay=1e-3)

score, preds = model_4.score(validation_dataset)
plot_confusion_matrix(confusion_matrix(y_dev,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_4 Confusion Matrix on Validation Set',
                      cmap=plt.cm.Blues)
print(f"Model_4 Accuracy on Validation Set is {score}")

score, preds = model_4.score(test_dataset)
plot_confusion_matrix(confusion_matrix(y_test,preds),
                      classes = [0,1,2,3,4,5,6,7,8,9],
                      normalize=False,
                      title='Model_4 Confusion Matrix on Test Set',
                      cmap=plt.cm.Reds)
print(f"Model_4 Accuracy on Test Set is {score}")

plt.plot(epochs, train_loss, "green")
plt.plot(epochs, val_loss, "blue")
plt.grid(True)
plt.title("Model_4 Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
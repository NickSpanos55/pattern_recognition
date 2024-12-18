from sklearn.datasets import load_files
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#-ΣΗΜΕΙΩΣΗ : Οι εντολές για την εμφάνιση των plots έχουν γίνει comments. 
#-Τα plots είναι τοποθετημένα στην αναφορά. Αυτό έγινε γιατί η matplotlib
#-αφήνει δύο plots ενεργά κάθε φορά και σταματάει την ροή του κώδικα.

#Μέλη Ομάδας 
#Νικόλαος Χάιδος - 03118096
#Νικόλαος Σπανός - 03118822

#-------------- Step 1 --------------#

#-Loading train and test datasets-#

train = np.loadtxt('./data/train.txt')
test = np.loadtxt('./data/test.txt')

#-Dividing to Labels and Features-#

X_train = train[:,1:]
X_test = test[:,1:]
y_train = train[:,0]
y_test = test[:,0]

#-------------- Step 2 --------------#

#-Reshaping and showing digit number 131('0')-#

digit = X_train[130]
digit = digit.reshape((16,16))
plt.title("Step 2 - Digit 0")
#plt.imshow(digit,cmap = 'gray')


#-------------- Step 3 --------------#

#-Showing one random sample from each digit-#

fig = plt.figure()
fig.suptitle('Step 3 - All Digits')
for i in range(0,10):
  temp = train[train[:,0] == i,1:257]
  index = np.random.choice(temp.shape[0],1)
  digit = temp[index]
  digit = digit.reshape((16,16))
  ax = fig.add_subplot(2,5,i+1)
  ax.imshow(digit,cmap = 'gray')
#plt.show()



#-------------- Step 4-5 --------------#


#--Computing Mean and Variance for Pixel (10,10) for digit '0'--#

temp = train[train[:,0] == 0.,1:257]
temp = temp.reshape(temp.shape[0],16,16)

final_mean = np.mean(temp[:,9,9])
final_var = np.var(temp[:,9,9])
print("#-------------- Step 4-5 --------------#",'\n')
print(f"Final mean: {final_mean}\n\n")
print(f"Final var: {final_var}\n\n")


#-------------- Step 6 --------------#


#--Computing Mean and Variance for All Pixels for digit '0'--#
temp = train[train[:,0] == 0.,1:257]

final_mean = np.mean(temp, axis = 0)
final_var = np.var(temp, axis = 0)



#-------------- Step 7 --------------#

#-Showing Means for all pixels for digit '0'-#
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.imshow(final_mean.reshape(16,16))
ax.set_title("Digit 0 Pixel Means - Step 7")


#-------------- Step 8 --------------#

#-Showing Variances for all pixels for digit '0'-#
ax = fig.add_subplot(1,2,2)
ax.imshow(final_var.reshape(16,16))
ax.set_title("Digit 0 Pixel Variances - Step 8")
#fig.show()

#-------------- Step 9a --------------#

#-Calculating Means and Variances for all pixels of all digits-#
final_means = np.zeros(shape = (10,256))
final_vars = np.zeros(shape = (10,256))

for i in range(0,10):
  temp = train[train[:,0] == i,1:257]

  final_means[i] = np.mean(temp, axis = 0)
  final_vars[i] = np.var(temp, axis = 0)


#-------------- Step 9b --------------#

#-Showing Means for all pixels for all digits-#
fig = plt.figure()
fig.suptitle('All Digit Means - Step 9')
for i in range(0,10):
  digit = final_means[i]
  digit = digit.reshape((16,16))
  ax = fig.add_subplot(2,5,i+1)
  ax.imshow(digit,cmap = 'gray')
#plt.show()


#-------------- Step 10 --------------#

#-Showing Digit 101 of Test Data-#
digit = X_test[100]
digit = digit.reshape((16,16))
#plt.imshow(digit,cmap = 'gray')
plt.title("Step 10 - Digit 101 of Test Data")


#-Predicting the digit's class with Euclidean Distance between class Means and sample pixels-#
digit = X_test[100]
dist = np.linalg.norm(final_means - digit, axis = 1, ord = 2)       # Order-2 Norm is equal to Euclidean Distance
pred_class = np.argmin(dist)
print("#-------------- Step 10 --------------#",'\n')
print("Predicted ClassΑ: " , pred_class,'\n')
print("True Class: ", int(y_test[100]),'\n','\n')


#-------------- Step 11 --------------#

#-Calculating accuracy of prediction for all samples-#
final_preds = np.zeros(shape = (y_test.shape[0],))

for i in range(X_test.shape[0]):
  dist = np.linalg.norm(final_means - X_test[i], axis = 1, ord = 2)
  final_preds[i] = np.argmin(dist)

acc = (final_preds == y_test).sum() / (y_test.shape[0])

print("#-------------- Step 11 --------------#",'\n')
print(f"Accuracy: {100*acc:.3f}% \n \n")

#-------------- Step 12 --------------#

#-Creating Euclidean Distance Classifier in accordance to sklearn-#
def euclidean_distance_classifier(X, X_mean):
    final_preds = np.zeros(shape = (X.shape[0],))

    for i in range(X.shape[0]):
      dists = np.linalg.norm(X_mean - X[i], axis = 1, ord = 2)
      final_preds[i] = np.argmin(dists)

    return final_preds


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        self.X_mean_ = np.zeros(shape = (10,X.shape[1]))
        for i in range(0,10):
            temp = X[y == i,:]
            self.X_mean_[i] = np.mean(temp, axis = 0)
        
        return self
    
    def predict(self, X):
        return euclidean_distance_classifier(X, self.X_mean_)
       

    def score(self, X, y):
        acc = (self.predict(X) == y).sum() / (y.shape[0])
        return acc


#-------------- Step 13a --------------#

#-Calculating 5-Fold Cross Validation Score for the Custom Classifier-#
clf = EuclideanDistanceClassifier()

cv_score = cross_val_score(clf, X_train, y_train, cv=KFold(n_splits=5), scoring="accuracy")

print("#-------------- Step 13a --------------#",'\n')
print(f"Cross Validation Score: {100*np.mean(cv_score):.3f}% \n \n")



#-------------- Step 13b --------------#

#-Plotting Decision Boundary for the Euclidean Classifier-#
def plot_clf(clf, X, y, labels):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of Classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    for digit in range(10):
        ax.scatter(X0[y == digit], X1[y == digit], label=labels[digit], 
                    s=60, alpha=0.9, edgecolors='k')
    
    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.title("Step 13b - Decision Surface")
    plt.show()



#-Reducing Dimensionality using PCA to draw a 2-D plot-#

pca = PCA(n_components=2)
X_new = pca.fit_transform(X_train)

clf.fit(X_new, y_train)

#plot_clf(clf, X_new, y_train, [i for i in range(10)])


#-------------- Step 13c --------------#

from sklearn.model_selection import learning_curve

#-Drawing Learning Curve for the Classifier-#

train_sizes, train_scores, test_scores = learning_curve(
    EuclideanDistanceClassifier(), X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5))



def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.title("Step 13c - Learning Curve")
    plt.show()

#plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.6, 1))


#-------------- Step 14 --------------#


#-Calculating prior probabilities for Naive Bayes Classifier-#

priors = np.unique(y_train.astype(int), return_counts=True)[1] / y_train.shape[0]


#-------------- Step 15a --------------#

#-Creating Custom Naive Bayes Classifier in accordance to sklearn-#
class CustomNBClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, use_unit_variance=False):
        #-Initialization of required parameters-#
        self.use_unit_variance = use_unit_variance
        self.final_means = None
        self.final_vars = None
        self.priors = None
        self.variance_zero_check = None


    def fit(self, X, y):
        self.final_means = np.zeros(shape = (10,X.shape[1]))
        self.final_vars = np.zeros(shape = (10,X.shape[1]))

        for i in range(0,10):
            temp = X[y == i,:]
            #-Calculating Means and Variances according to data for fitting-#
            self.final_means[i] = np.mean(temp, axis = 0)
            self.final_vars[i] = np.var(temp, axis = 0)
        
        if self.use_unit_variance:
            self.final_vars = np.ones(self.final_vars.shape)

        self.priors = np.unique(y.astype(int), return_counts=True)[1] / y.shape[0]
        
        
        return self


    def predict(self, X):
        posterior = np.empty((X.shape[0],10))

        #-Calculating Mutlivariate Normal Distribution for each pixel of all digits for posterior probabilities-#
        for digit in range(10):

            #-Calculating Posterior Probabilities by multiplying distribution with priors-#
            posterior[:, digit] = scipy.stats.multivariate_normal(self.final_means[digit,:], self.final_vars[digit,:], 
                                                                allow_singular=True).pdf(X[:, :]) * self.priors[digit]

        #-Selecting maximum digit probability for final prediction-#
        preds = np.argmax(posterior, axis=1)
        
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        acc = (self.predict(X) == y).sum() / (y.shape[0])
        
        return acc


#-------------- Step 15b-c --------------#

#-Calculating Custom Naive Bayes Score and sklearn Naive Bayes Score for comparison-# 
clf_custom_nb = CustomNBClassifier()
clf_custom_nb.fit(X_train, y_train)

print("#-------------- Step 15b-c --------------#",'\n')
print(f"Custom Naive Bayes Score: {100*clf_custom_nb.score(X_test, y_test):.3f}%")



from sklearn.naive_bayes import GaussianNB

clf_sklearn_nb = GaussianNB()
clf_sklearn_nb.fit(X_train, y_train)

print(f"SKlearn Naive Bayes Score: {100*clf_sklearn_nb.score(X_test, y_test):.3f}% \n")


#-------------- Step 16--------------#

#-Calculating Custom Naive Bayes Classifier with unit variance matrix-#
clf_custom_nb = CustomNBClassifier(use_unit_variance=True)
clf_custom_nb.fit(X_train, y_train)

print("#-------------- Step 16 --------------#",'\n')
print(f"Custom Naive Bayes(Unit Variance) Score: {100*clf_custom_nb.score(X_test, y_test):.3f}% \n")


#-------------- Step 17 --------------#

#-Comparing Different Classifier Implementations(KNN,SVM,Naive Bayes)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

# Classifiers Testing

clf_sklearn_nb = GaussianNB()
clf_sklearn_nb.fit(X_train, y_train)
nb_score = clf_sklearn_nb.score(X_test, y_test)

clf_sklearn_knn3 = KNN(n_neighbors=3)
clf_sklearn_knn3.fit(X_train, y_train)
knn3_score = clf_sklearn_knn3.score(X_test, y_test)

clf_sklearn_knn5 = KNN(n_neighbors=5)
clf_sklearn_knn5.fit(X_train, y_train)
knn5_score = clf_sklearn_knn5.score(X_test, y_test)

clf_sklearn_knn7 = KNN(n_neighbors=7)
clf_sklearn_knn7.fit(X_train, y_train)
knn7_score = clf_sklearn_knn7.score(X_test, y_test)

clf_sklearn_svm_linear = SVC(kernel='linear')
clf_sklearn_svm_linear.fit(X_train, y_train)
svm_linear_score = clf_sklearn_svm_linear.score(X_test, y_test)

clf_sklearn_svm_poly = SVC(kernel='poly')
clf_sklearn_svm_poly.fit(X_train, y_train)
svm_poly_score = clf_sklearn_svm_poly.score(X_test, y_test)

clf_sklearn_svm_rbf = SVC(kernel='rbf')
clf_sklearn_svm_rbf.fit(X_train, y_train)
svm_rbf_score = clf_sklearn_svm_rbf.score(X_test, y_test)

print("#-------------- Step 17 --------------#",'\n')
print(f"SKlearn Naive Bayes Score: {100*nb_score:.3f}%")
print(f"SKlearn 3-NN Score: {100*knn3_score:.3f}%")
print(f"SKlearn 5-NN Score: {100*knn5_score:.3f}%")
print(f"SKlearn 7-NN Score: {100*knn7_score:.3f}%")
print(f"SKlearn Linear SVM Score: {100*svm_linear_score:.3f}%")
print(f"SKlearn Poly SVM Score: {100*svm_poly_score:.3f}%")
print(f"SKlearn RBF SVM Score: {100*svm_rbf_score:.3f}% \n")


#-------------- Step 18a --------------#

#-Testing Ensembling Methods(VottingClassifier) for the best classifiers from previous step-#
from sklearn.ensemble import VotingClassifier

voting_hard_clf = VotingClassifier(estimators = [('cnb',CustomNBClassifier()),('sknb',GaussianNB()),
                                              ('KNN3',KNN(n_neighbors=3)),('KNN5',KNN(n_neighbors=5)),
                                              ('PolySVM',SVC(kernel='poly'))],voting = 'hard',n_jobs=-1)
voting_hard_clf.fit(X_train,y_train)
voting_hard_score = voting_hard_clf.score(X_test,y_test)

print("#-------------- Step 18a --------------#",'\n')
print(f"Voting Classifier(Hard) Score: {100*voting_hard_score:.3f}% \n")


#-------------- Step 18b --------------#

#-Testing Ensembling Methods(BagginClassifier) for 20 estimators of Custom Naive Bayes-#
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(base_estimator=CustomNBClassifier(),n_estimators=20).fit(X_train, y_train)

bagging_score = bagging_clf.score(X_test,y_test)

print("#-------------- Step 18b --------------#",'\n')
print(f"Bagging Classifier Score: {100*bagging_score:.3f}% \n")


#-------------- Step 19 --------------#

#-Implementation of Neural Network with Pytorch for Digit Recognition-#
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDigitDataset(Dataset):
    def __init__(self, ndarray):
        self.table = torch.from_numpy(ndarray)

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        return self.table[idx, 1:], self.table[idx, 0].long()

#-Loading train and test datasets-#
train_ndarray = np.loadtxt("./data/train.txt", dtype="float32")
test_ndarray = np.loadtxt("./data/test.txt", dtype="float32")

#-Split into validation and train set(80-20)-#
val_index = int(train_ndarray.shape[0] * 0.8)

train_data = CustomDigitDataset(train_ndarray[:val_index, :])
validation_data = CustomDigitDataset(train_ndarray[val_index:, :])
test_data = CustomDigitDataset(test_ndarray)



#-Creating Pytorch DataLoader with batch size = 16-#
train_dataloader = DataLoader(train_data, batch_size=16)
validation_dataloader = DataLoader(validation_data, batch_size=16)
test_dataloader = DataLoader(test_data, batch_size=16)



print("#-------------- Step 19 --------------#",'\n')
#-Creating Custom Neural Network as subclass of nn.Module-#
class CustomNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #-3 Hidden Layers((300,200,100))
        self.layer1 = torch.nn.Linear(256, 300)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(300, 200)
        self.relu2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(200, 100)
        self.relu3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(100, 10)
        #-Last Layer Softmax Activation Function for probability Distribution-#
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        output = self.relu1(self.layer1(x))
        output = self.relu2(self.layer2(output))
        output = self.relu3(self.layer3(output))
        output = self.softmax(self.layer4(output))
        return output
    
    def predict(self, x):
        output = self.relu1(self.layer1(x))
        output = self.relu2(self.layer2(output))
        output = self.relu3(self.layer3(output))
        output = self.softmax(self.layer4(output))
        return output
    
    def score(self, x, y):
        preds = model.predict(x)
        preds = torch.argmax(preds, dim=1)
        acc = (y.detach().numpy() == preds.detach().numpy())
        acc = acc.sum() / acc.shape[0]
        return acc
    
    def fit(self, EPOCHS=30):
        #-Function for random weight initialization-#
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)
        #-Initializing entropy loss and optimizer(AdamW) and learning rate scheduler-#
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters())
        
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        #-Creating function for training epochs and computing average loss for every 100 batches-#
        def train_one_epoch(epoch_index):
            running_loss = 0.0
            last_loss = 0.0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                # initialize weight gradients back to zero
                optimizer.zero_grad()

                # gradients change for one forward pass (one batch)
                outputs = self.forward(inputs)

                # calculate loss and change weights
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100
                    running_loss = 0.0

            return last_loss

        for epoch in range(EPOCHS):

            self.train(True)
            avg_loss = train_one_epoch(epoch)
            self.train(False)                       # disable training for validation data scoring

            #-Calculating train and validation set losses for each epoch-#
            running_vloss = 0.0
            for i, vdata in enumerate(validation_dataloader):
                vinputs, vlabels = vdata
                voutputs = self.forward(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('EPOCH {}, Train Loss {:.3f}, Validation Loss {:.3f}'.format(epoch+1, avg_loss, avg_vloss))
        

#-Fitting Neural Network and Calculating Score on test set-#
model  = CustomNeuralNet()
model.fit(30)
model_score = model.score(torch.tensor(test_ndarray[:, 1:]), torch.tensor(test_ndarray[:, 0]))

print(f"\nNeural Network Score: {100*model_score:.3f}% \n")
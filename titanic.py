import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import tree
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#libs for deep nn
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score


#data import
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#describe data set
print("\n\nDescribe\n")
print(train.describe())
#top 5 rows
print('\n\nTop 5 values\n')
print(train.head(5))
#num of nulls
print('\n\nNumber of nulls\n')
print(train.isnull().sum())
#Survived value counts
print('\n\nSurvived value counts\n')
print(train.Survived.value_counts())
#pclass value counts
print('\n\nPclass value counts\n')
print(train.Pclass.value_counts())
#sex value counts
print('\n\nSex value counts\n')
print(train.Sex.value_counts())
#Cabin value counts
print('\n\nCabin value counts\n')
print(train.Cabin.value_counts())

#dtypes
print('\n\nData Types of columns\n')
print(train.dtypes)

#death cases by sex
#male
print('Male Survived value counts')
#death count
maleDeath = train['Survived'][train['Sex']=='male'].value_counts()
print(maleDeath)
#death rate
print('\n\nMale death rate: ')
print(str(round(468/maleDeath.sum(),4)*100)+'%')
#plotting out results for male
plt.bar([1,0],height=maleDeath,color=['green','red'],tick_label=[0,1])
plt.show()

#female
print('\n\nFemale Survived value counts')
#death count
femaleDeath = train['Survived'][train['Sex']=='female'].value_counts()
print(femaleDeath)
#death rate
print('\n\nFemale death rate: ')
print(str(round(81/femaleDeath.sum(),4)*100)+'%')
#plotting out results for female
plt.bar([0,1], height=femaleDeath, color=['red','green'], tick_label=[1,0])
plt.show()

#where embarked is null
print('\n\nPassengers with null embarked:')
print(train[train['Embarked'].isna()])

#taking data i will use in first approach from train and test data set
#columns list
columnsToUseV1 = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#train Data set X I have pllied Survived for corr heatmap
#deep copy to avoid errors
trainSelectedV1 = train[columnsToUseV1+['Survived']].copy(deep=True)
#test data set
submitSelectedV1 = test[columnsToUseV1].copy(deep=True)

#checking nulls again
print('\n\nV1 missing values in train')
print(trainSelectedV1.isnull().sum())
print(trainSelectedV1.describe(include='object'))
print('\n\nV1 missing values in submit')
print(submitSelectedV1.isnull().sum())
print(submitSelectedV1.describe(include='object'))


#filling null's
#age in train
trainSelectedV1.Age.fillna(trainSelectedV1.Age.mean(),inplace=True,axis=0)
#dropping na from embarked
trainSelectedV1.dropna(inplace=True)

#checking how filling null's and dropping null's went
print('\n\nV1 missing values in train')
print(trainSelectedV1.isnull().sum())
print(trainSelectedV1.describe(include='object'))

#filling null's in submit
submitSelectedV1.Age.fillna(trainSelectedV1.Age.mean(),inplace=True,axis=0)
submitSelectedV1.Fare.fillna(trainSelectedV1.Fare.mean(),inplace=True,axis=0)

#checking how filling null's went
print('\n\nV1 missing values in submit')
print(submitSelectedV1.isnull().sum())
print(submitSelectedV1.describe(include='object'))

#I will translate sex and embarked to include the in correlation heatmap
trainSelectedV1['Sex'] = trainSelectedV1['Sex'].apply(lambda x: 0 if x == 'male' else 1)
trainSelectedV1.loc[trainSelectedV1['Embarked'] == 'C', 'Embarked'] = 0
trainSelectedV1.loc[trainSelectedV1['Embarked'] == 'Q', 'Embarked'] = 1
trainSelectedV1.loc[trainSelectedV1['Embarked'] == 'S', 'Embarked'] = 2
trainSelectedV1.Embarked = trainSelectedV1.Embarked.apply(pd.to_numeric)

#applying transformations to submit data set
submitSelectedV1['Sex'] = submitSelectedV1['Sex'].apply(lambda x: 0 if x == 'male' else 1)
submitSelectedV1.loc[submitSelectedV1['Embarked'] == 'C', 'Embarked'] = 0
submitSelectedV1.loc[submitSelectedV1['Embarked'] == 'Q', 'Embarked'] = 1
submitSelectedV1.loc[submitSelectedV1['Embarked'] == 'S', 'Embarked'] = 2
submitSelectedV1.Embarked = submitSelectedV1.Embarked.apply(pd.to_numeric)

#correlation heatmap
cormat = trainSelectedV1.corr()
sns.heatmap(cormat,cmap="coolwarm")

trainSelectedV1Normalized = trainSelectedV1
submitSelectedV1Normalized = submitSelectedV1

#distribution of features
sns.displot(trainSelectedV1Normalized, x="Fare")
sns.displot(trainSelectedV1Normalized, x="Age")
#Fare have high skewness (to the right)
#Im going to drop rows where Fare>300
#checking how much of those(fare>300) passengers we have in data set
print(trainSelectedV1Normalized[trainSelectedV1Normalized['Fare']>300])
#checking how much of those we have in test data set
print(submitSelectedV1Normalized[submitSelectedV1Normalized['Fare']>300])
#dropping those values
toDrop = trainSelectedV1[trainSelectedV1['Fare']>300].index
trainSelectedV1.drop(toDrop,axis=0, inplace=True)
#making sure if values were drop properly
print(trainSelectedV1Normalized[trainSelectedV1Normalized['Fare']>300])
#plotting out new Fare distribution
sns.displot(trainSelectedV1Normalized, x="Fare")

#scaler definition
scaler = StandardScaler()

#standard scaling of Age
scaler.fit(trainSelectedV1Normalized['Age'].values.reshape(-1,1))
trainSelectedV1Normalized['Age'] = scaler.transform(trainSelectedV1Normalized['Age'].values.reshape(-1,1))
submitSelectedV1Normalized['Age'] = scaler.transform(submitSelectedV1Normalized['Age'].values.reshape(-1,1))

#standard scaling of Fare
scaler.fit(trainSelectedV1Normalized['Fare'].values.reshape(-1,1))
trainSelectedV1Normalized['Fare'] = scaler.transform(trainSelectedV1Normalized['Fare'].values.reshape(-1,1))
submitSelectedV1Normalized['Fare'] = scaler.transform(submitSelectedV1Normalized['Fare'].values.reshape(-1,1))

#creating labels var
y = trainSelectedV1Normalized.Survived

#dropping Survived
trainSelectedV1Normalized.drop(labels='Survived', inplace=True,axis=1)
print(trainSelectedV1Normalized.head(5))

#to numpy
trainNp = trainSelectedV1Normalized.to_numpy()
submitNp = submitSelectedV1Normalized.to_numpy()
labelsNp = y.to_numpy()

#train test split
trainX, valX, trainY, valY = train_test_split(trainNp, labelsNp, test_size= 0.2)

#no hiperparamter tunning
logReg = LogisticRegression()
logReg.fit(trainX, trainY)
print(logReg.score(valX, valY))

#with some of the solvers l1 penalty cannot be computed, but GridSearchCv omit them and provide best results of computable options
parameters = {'C':[0.1,0.3,0.5,0.8,1,3,5,10,50,100],'penalty':['l1','l2'], 'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}

gsCV = GridSearchCV(logReg, parameters, cv=10)
gsCV.fit(trainX, trainY)

#print(gsCV.best_score_)
#print("\n\n", gsCV.best_params_)
bestParams = gsCV.best_params_

logReg = LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'],solver=bestParams['solver'])
logReg.fit(trainX, trainY)
print(logReg.score(valX, valY))

sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = logReg.predict(submitNp)


sub.to_csv('subCsvLogReg.csv',index=False)


#no hiperparamter tunning
svm = SVC()
svm.fit(trainX, trainY)
print(svm.score(valX, valY))

#due to the fact that it is on the border of range I will provide another grid
parameters = {'C':[0.1,0.3,0.5,0.8,1,3,5,10,50,100,200],'kernel':['rbf','poly', 'linear', 'sigmoid']}

gsCV = GridSearchCV(svm, parameters, cv=5)
gsCV.fit(trainX, trainY)

#print(gsCV.best_score_)
#print("\n\n", gsCV.best_params_)
bestParams = gsCV.best_params_

svm = SVC(C = bestParams['C'],kernel = bestParams['kernel'])
svm.fit(trainX, trainY)
print(svm.score(valX, valY))

sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = svm.predict(submitNp)

sub.to_csv('subCsvSVM.csv',index=False)

#no hiperparamter tunning

decTree = tree.DecisionTreeClassifier()
decTree.fit(trainX, trainY)
print(decTree.score(valX, valY))


sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = decTree.predict(submitNp)

sub.to_csv('subCsvtree.csv',index=False)

#due to the fact that it is on the border of range I will provide another grid
parameters = {'criterion':['gini', 'entropy'], 'max_depth':[3,4,5,6,7,8,9,10], 'splitter':['best','random'], 'min_samples_split':[2,3,4,5,6]}
gsCV = GridSearchCV(decTree, parameters, cv=10)
gsCV.fit(trainX, trainY)
print(gsCV.best_score_)
print("\n\n", gsCV.best_params_)
bestParams = gsCV.best_params_

decTree = tree.DecisionTreeClassifier(criterion = bestParams['criterion'], max_depth=bestParams['max_depth'], splitter = bestParams['splitter'], min_samples_split = bestParams['min_samples_split'])
decTree.fit(trainX, trainY)
print(decTree.score(valX, valY))

sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = decTree.predict(submitNp)

sub.to_csv('subCsvtree.csv',index=False)

#no hiperparamter tunning

xgbClass = XGBClassifier()
xgbClass.fit(trainX, trainY)
print(xgbClass.score(valX, valY))

sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = xgbClass.predict(submitNp)

sub.to_csv('subCsvXgb.csv',index=False)

parameters = {'booster':['gbtree','dart'], 'gamma':[0.01,0.05,0.1,0.5,1,5,10],'max_depth':[4,5,6,7,8,9,10]}

gsCV = GridSearchCV(xgbClass, parameters, cv=2, n_jobs=-1)
gsCV.fit(trainX, trainY)

print(gsCV.best_score_)
print("\n\n", gsCV.best_params_)
bestParams = gsCV.best_params_

xgbClass = XGBClassifier(booster = bestParams['booster'], gamma = bestParams['gamma'],max_depth = bestParams['max_depth'])
xgbClass.fit(trainX, trainY)
print(xgbClass.score(valX, valY))
sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = xgbClass.predict(submitNp)
sub.to_csv('subCsvtree.csv',index=False)

votingHard = VotingClassifier(estimators=[('lr', logReg), ('svc', svm), ('tree', decTree), ('xgb', xgbClass)], voting='hard')
votingSoft = VotingClassifier(estimators=[('lr', logReg), ('svc', svm), ('tree', decTree), ('xgb', xgbClass)], voting='soft')

votingHard.fit(trainX,trainY)
#votingSoft.fit(trainX,trainY)

print(votingHard.score(valX,valY))
#print(votingSoft.score(valX,valY))

sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = votingHard.predict(submitNp)
sub.to_csv('subCsvVoteHard.csv',index=False)


#chceking if cuda is available. Setting correct device
if torch.cuda.is_available():
    print('Cuda')
    dev = "cuda"
else:
    print('CPU')
    dev = "cpu"
device = torch.device(dev)

trainXtens = torch.from_numpy(trainX).to(torch.float32)
trainYtens = torch.from_numpy(trainY).to(torch.float32)
valXtens = torch.from_numpy(valX).to(torch.float32)
valYtens = torch.from_numpy(valY).to(torch.float32)
subTens = torch.from_numpy(submitNp).to(torch.float32)

trainset = TensorDataset(trainXtens,trainYtens)

trainloader = DataLoader(trainset,batch_size=5,shuffle=False)

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

inputSize = 7
model = Net(inputSize)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
lossFn = nn.BCELoss()

epochs = 50

losses = []
accur = []
for i in range(epochs):
    for j,(xTrain,yTrain) in enumerate(trainloader):
        #calculate output
        output = model(xTrain)
        #calculate loss
        loss = lossFn(output,yTrain.reshape(-1,1))
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predAll = model(valXtens)
    #accuracy
    accmat = np.where(predAll <= 0.5, 0, predAll.detach().numpy())
    accmat = np.where(accmat > 0.5, 1, accmat)
    acc = accuracy_score(valYtens, accmat)
    losses.append(loss)
    accur.append(acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

predAll = model(subTens)
resmat = np.where(predAll <= 0.5, 0, predAll.detach().numpy())
resmat = np.where(resmat > 0.5, 1, resmat)


sub=pd.read_csv('gender_submission.csv')
sub['Survived'] = resmat.astype(int)
sub.to_csv('subCsvNN4.csv',index=False)
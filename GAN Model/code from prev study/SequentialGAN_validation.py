#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
get_ipython().run_line_magic('matplotlib', 'inline')
sb.set()

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc as getAuc,roc_auc_score
from sklearn.metrics import confusion_matrix
import itertools


# ## Load original Data for using as Test Set##

# In[ ]:


TEST_FILE = os.path.join(os.path.abspath("."),'..', "Data","quartiles_test.csv")
testData = pd.read_csv(TEST_FILE)
testData.head()


# In[ ]:


Y_test=testData['finalResult'].copy()
X_test=testData.drop(['finalResult'], axis=1)
print('Testing Set Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)
features_list=list(X_test.columns)
print(features_list)


# In[ ]:


TRAIN_FILE = os.path.join(os.path.abspath("."),'..', "Data",'quartiles_train.csv')
trainData = pd.read_csv(TRAIN_FILE)
trainData.head()


# In[ ]:


Y_train=trainData['finalResult'].copy()
X_train=trainData.drop(['finalResult'], axis=1)
print('Training Set Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)


# In[ ]:


#Visualization Function#
def plot_ROC(data,probs,algo):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(data, probs,pos_label='Pass')
    roc_auc = getAuc(fpr, tpr)
    plt.figure(figsize = (5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example "' + algo+'"',size = 24)
    plt.legend(loc="lower right")
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# In[ ]:


def loadModel(filename,weights=False):
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    if weights:
        # load weights into new model
        loaded_model.load_weights(filename+".h5")
    
    loaded_model.compile( optimizer=optimizers.Adagrad(lr=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
    return loaded_model


# In[ ]:


algoResultsDf=pd.DataFrame(columns=['classifier','algorithm','trainingAccuracy','testingAccuracy','precision','recall','f1-score','auc'])
def trainAndVisiualize(X_train,Y_train,algo,random=100):
    def saveScoresInDataFrame(classifierName,classifier,trainPredictions,testPredictions):
        trainingScore=accuracy_score(y_true = Y_train, y_pred = trainPredictions)
        testingScore=accuracy_score(y_true = Y_test, y_pred = testPredictions)

        print('Training Accuracy = ',trainingScore)
        print('Testing Accuracy = ',testingScore)

        # Probabilities for each class
        if classifierName=='RandomForest':
            rf_probs = classifier.predict_proba(X_test)[:,1]
        else:
            rf_probs = classifier.predict_proba(X_test)
            
        roc_value = roc_auc_score(Y_test, rf_probs)
        plot_ROC(Y_test,rf_probs,algo)
        # Confusion matrix
        cm = confusion_matrix(Y_test, testPredictions,labels=['Pass', 'Fail'])
        plot_confusion_matrix(cm, classes = ['Pass', 'Fail'],title='Confusion Matrix "' + classifierName+' - '+ algo+'"')


        tn=cm[0,0]
        tp=cm[1,1]
        fn=cm[1,0]
        fp=cm[0,1]
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1=2*(precision*recall) / (precision+recall)

        algoResultsDf.loc[len(algoResultsDf)] = [classifierName,algo,trainingScore,testingScore,precision,recall,f1,roc_value]
        
    print('Training Set Shape:', X_train.shape)
    print('Training Labels Shape:', Y_train.shape)
    
    classifier=RandomForestClassifier(
        n_estimators = 500,
        random_state = 0,
        max_depth=9,
        max_features=6,
        min_samples_split=4,
        class_weight={'Pass': 1, 'Fail': 2},
        min_impurity_decrease=1e-6
    )
    
    classifier.fit(X_train, Y_train);
    trainPredictions = classifier.predict(X_train)
    testPredictions = classifier.predict(X_test)
    
    saveScoresInDataFrame('RandomForest',classifier,trainPredictions,testPredictions)
    
    nnClassifier=loadModel('../NN_Classifier/quartiles_classifier')
    
    Y_train_int,yMap=pd.factorize(Y_train,sort=True)
    print(yMap)
    nnClassifier.fit(X_train,Y_train_int,  validation_split=0.1,epochs=50, batch_size=128,verbose=0)
    trainPredictions = nnClassifier.predict_classes(X_train).reshape(-1)
    trainPredictions=np.array(list('Pass' if i==1 else 'Fail' for i in trainPredictions))
    testPredictions = nnClassifier.predict_classes(X_test).reshape(-1)
    testPredictions=np.array(list('Pass' if i==1 else 'Fail' for i in testPredictions))
    saveScoresInDataFrame('NeuralNetwork',nnClassifier,trainPredictions,testPredictions)
    
    return


# In[ ]:


def compareResults(classifier,trainingAccs,testingAccs,auc,f1,labels):
    # data to plot
    fig, ax = plt.subplots(figsize=(30,7))
    n_groups=len(trainingAccs)
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, trainingAccs, bar_width,
    alpha=opacity,
    color='b',
    label='Training Accuracy')

    rects2 = plt.bar(index + bar_width, testingAccs, bar_width,
    alpha=opacity,
    color='g',
    label='Testing Accuracy')
    
    rects3 = plt.bar(index + 2*bar_width, auc, bar_width,
    alpha=opacity,
    color='r',
    label='ROC-AUC')
    
    rects4 = plt.bar(index + 3*bar_width, f1, bar_width,
    alpha=opacity,
    color='y',
    label='F1-Score')

    plt.xlabel('Oversampling Algorithm',fontsize=15)
    plt.ylabel('Value',fontsize=15)
    plt.ylim(0.73,1)
    plt.title('Oversampling algo comparisons for OULAD ('+classifier+')',fontsize=20)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plotPrec_Recall(classifier,precision,recall,labels):
    # data to plot
    fig, ax = plt.subplots(figsize=(30,7))
    n_groups=len(labels)
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 0.8

    rects1 = plt.bar(index, precision, bar_width,
    alpha=opacity,
    color='b',
    label='Precision')

    rects2 = plt.bar(index + bar_width, recall, bar_width,
    alpha=opacity,
    color='r',
    label='Recall')

    plt.xlabel('Oversampling Algorithm',fontsize=15)
    plt.ylabel('Value',fontsize=15)
    plt.ylim(0.65,1)
    plt.title('Oversampling algorithms Precision-Recall comparisons for OULAD ('+classifier+')',fontsize=20)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[ ]:


# Transform data into sequential quartiles#
columns=[c.strip() for c in  ['finalResult']+features_list ] 
print(columns)


# In[ ]:


def concatenateSamples(quartiles,toSample,condition=True):
    df=quartiles[0]  # first quartile labels will be used in concatenated data
    if condition:
        df=df[(df.finalResult==r) &  (df.codeModule==c) & (df.moduleSession==m)]
        
    original=df.sample(toSample).copy().reset_index(drop=True)
    temp=original.iloc[:,0:3]
    originalF=original.iloc[:,3:]

    q=2
    for synth in quartiles[1:]:
        if condition:
            synth=synth[(synth.finalResult==r) & (synth.codeModule==c) & (synth.moduleSession==m)]            
        synthF=synth.sample(toSample).copy().reset_index(drop=True)
        synthF=synthF.iloc[:,3:]
        synthF.columns=[ 'Q'+str(q)+' '+col for col in synthF.columns ]
        q+=1
        temp=pd.concat([temp,synthF],axis=1)

    originalF.columns=[ 'Q1 '+col for col in originalF.columns]
    
    return pd.concat([temp,originalF],axis=1)
    
def getConcatenatedQuartiles(quartiles,uniform=False):
    toSample=950   # total 28 classes so it will be around 13500
    df = None

    for i in range(len(quartiles)):
        labels,a = pd.factorize(quartiles[i]['codeModule'])
        quartiles[i].codeModule=labels
        labels,a = pd.factorize(quartiles[i]['moduleSession'])
        quartiles[i].moduleSession=labels

    data=pd.DataFrame()
    for r in ['Pass','Fail']:
        for c in range(0,7):
            for m in range(0,2):
                # for general case sample synthetic data doesn't contain all classes
                if not uniform:
                    toSample=10000
                    for q in quartiles:
                        toSample=min(toSample,q[(q.finalResult==r) &  (q.codeModule==c) & (q.moduleSession==m)].shape[0])
                
                temp=concatenateSamples(quartiles,toSample,False)
                data=pd.concat([data,temp],axis=0)
        
    # complete training set size
    if data.shape[0]<25000:
        toSample=25000-data.shape[0]
        temp=concatenateSamples(quartiles,toSample,False)
        data=pd.concat([data,temp],axis=0)
    
    return data.loc[:,columns]


# In[ ]:


PATH_DATA  = os.path.join(os.path.abspath("."),'..', "Sequential CGAN for each quartile")
q1 = pd.read_csv(PATH_DATA+'/CGAN Q1/Q1_synthetic.csv')
q2 = pd.read_csv(PATH_DATA+'/CGAN Q2/Q2_synthetic.csv')
q3 = pd.read_csv(PATH_DATA+'/CGAN Q3/Q3_synthetic.csv')
q4 = pd.read_csv(PATH_DATA+'/CGAN Q4/Q4_synthetic.csv')

CGANData=getConcatenatedQuartiles([q1,q2,q3,q4])


# In[ ]:


print(CGANData.shape)
CGANData.columns


# ## SC-GAN##

# In[ ]:


count=trainData['finalResult'].value_counts()
toGenerate=count[0]-count[1]


# In[ ]:


count=trainData['finalResult'].value_counts()
toGenerate=count[0]-count[1]

synthetic=CGANData[CGANData['finalResult']=='Fail']
combined=pd.concat([trainData,synthetic.sample(toGenerate)],sort=False)
y_res=combined['finalResult']
X_res=combined.drop(['finalResult'],axis=1)

print('Distribution after sequential CGAN:\t',pd.DataFrame(y_res,columns=['finalResult'])['finalResult'].value_counts())


# In[ ]:


algo='Sequential CGAN'
trainAndVisiualize(X_res,y_res,algo)
algoResultsDf


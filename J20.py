#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[131]:


import pandas as pd
import numpy as np
from scipy import stats
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler ,RobustScaler, Normalizer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_predict, GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, label_binarize, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.utils.multiclass import unique_labels

import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
from eli5 import explain_prediction_df

from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import HTML


# In[214]:


# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


# ## Functions

# In[3]:


def CV_best_params(param_grid, model, X_train, y_train):
    """Tuning the parameter of the model using Cross Validation"""
    grid_model = GridSearchCV(model, param_grid, cv=10)
    grid_model.fit(X_train,y_train)    
    return grid_model.best_params_


# In[177]:


def auc_multilabel(y_test, y_pred, n_classes = 4):

    y_test2 = label_binarize(y_test, classes=np.arange(n_classes))
    y_pred2 = label_binarize(y_pred, classes=np.arange(n_classes))
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_pred2[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test2.ravel(), y_pred2.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    return roc_auc


# In[172]:


def RandomForest(model, X_train, X_test, y_train, y_test, idx, title = '', figname = '', n_classes = 2):
    
    """ This function is used to solve the classification problem using Randon Forest Classifier
    
        Parameters:
            RandomForest(X,y,test_percentage, nt, md, mss)
            test_percentage: the percentage of total data used for testing the model
            nt: number of Trees, default=100
            md: min_depth, default=None
            mss: min_samples_split, default=2
        Outputs:
            y_pred: the predicted classes using test data
            acc: accuracy
            auc: area under the curve
            feature_imp: feature importance list
    """
    
#     Train the model
    model.fit(X_train, y_train)
    
#     Model Accuracy, how often is the classifier correct?
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    bacc = metrics.balanced_accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    print('Balanced Accuracy:', bacc)
    
    
#     Soft classification probabilities
    y_pred_proba = model.predict_proba(X_test)
    
#     Area under the ROC curve
    # Compute ROC curve and ROC area for each class
#     roc_auc = auc_multilabel(y_test, y_pred, n_classes)


#     false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
#     auc = metrics.auc(false_positive_rate, true_positive_rate)  # AUC: Area Under Curve
#     auc = metrics.roc_auc_score(y_test, y_pred)
#     auc = metrics.roc_auc_score(y_test, y_pred)
#     print('Area Under the ROC curve:', auc)
    
#     Feature importance
    feature_imp = pd.Series(model.feature_importances_,index=idx)
    
#     Visualizing features based on their importance
    plt.figure(figsize=(16,6.5))
    sns.barplot(y=feature_imp.sort_values(ascending=False)*100, x = feature_imp.sort_values(ascending=False).index)
    plt.xlabel('Features', fontsize = 15, fontweight='bold')
    plt.ylabel('FI Score (%)', fontsize = 15, fontweight='bold')
    plt.ylim([0,20])
    plt.xticks(fontsize = 13.5, rotation = 90)
    plt.yticks(fontsize = 13.5)
    plt.title(title, fontsize = 16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    
    return y_pred, y_pred_proba, acc, bacc, feature_imp, model.feature_importances_



def data_for_modeling(df, step):
    """This function is used to prepare train and test sets for each step in the test"""
    df_m = df.sort_values(by = ['Animal_ID','Test_Number'])
    df_m = df_m[df_m['Test_Number'] == step]
    X = np.array(df_m.iloc[:,6:])
    y = np.array(df_m.iloc[:,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


# In[6]:


def cross_val(model, X, y, folds = 20):
    """K-fold cross validation: the best approach if we have a limited input data"""
    print('Systematically create '+str(folds)+' train/test splits and fit the model to those data')
    cv_score = cross_val_score(model, X, y, cv = folds, scoring = 'accuracy')
    print('Cross validaton scores: ', '\n', cv_score)
    
    # Standard error of the mean of the scores
    print('Mean Score: ', cv_score.mean())
    print('Standard Error of the Mean Score: ', stats.sem(cv_score))

    # Plot histogram
    plt.hist(cv_score, bins = 4)
    plt.show()


# In[7]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title ='Confusion matrix, without normalization'
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm.shape)

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    # We want to show all ticks...
    
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylim = [max(np.arange(cm.shape[0]))+.5, -.5],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[178]:


def classification_metrics(model, X, y, labels, norm = False, title = None):
    """This function is used to prepare and present all metrics for the classification model"""
    
    # Predict the model
    y_pred = model.predict(X)
#     y_pred_proba = model.predict_proba(X)
    
    # Accuracy
    accuracy_rf = accuracy_score(y, y_pred, normalize = True)

    # Balanced accuracy
    balanced_accuracy_rf  = metrics.balanced_accuracy_score(y, y_pred)

    # Cohen's kappa
    cohens_kappa_rf = metrics.cohen_kappa_score(y,y_pred)
    
    # Area ubder the ROC curve (AUC)
    auc_rf = auc_multilabel(y, y_pred)
    
    print('*************************************************************')
    print('\n')
    print('Accuracy:', accuracy_rf)
    print('Balanced Accuracy: ', balanced_accuracy_rf)
    print('Cohens Kappa: ', cohens_kappa_rf)
    print('AUC (micro-averaged', auc_rf)
    
    print('\n')
    print('Classification report: ', '\n', )
    print('\n')
    
    # Plot confusion matrix
    plot_confusion_matrix(y, y_pred, labels,
                              normalize=norm,
                              title=title,
                              cmap=plt.cm.Blues)
    print('*************************************************************')


# In[9]:


def oob_classifier_accuracy(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    X = np.array(X_train)
    y = np.array(y_train)
    
    n_samples = len(X)
    n_classes = len(np.unique(y))
    predictions = np.zeros((n_samples, n_classes))
    
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict_proba(X[unsampled_indices])
        predictions[unsampled_indices] += tree_preds
        
    predicted_class_indexes = np.argmax(predictions, axis=1)
    predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]
    
    oob_score = np.mean(y == predicted_classes)
    return oob_score


# In[10]:


def permutation_importances(model, X_train, y_train, metric):
    ''' Return the permuation importance matrix in numpy array format'''
    baseline = metric(model,X_train,y_train)
    imp = []
    for col in X_train.columns:
        save = copy.deepcopy(X_train.iloc[:,col])
        X_train.iloc[:,col] = np.random.permutation(X_train.iloc[:,col])
        m = metric(model,X_train,y_train)
        X_train.iloc[:,col] = save
        imp.append(baseline - m)
    return np.array(imp)


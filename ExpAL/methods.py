import numpy as np
import random

from sklearn.utils import resample


def Random_Pick(X_pool, X_train = None, 
                y_train = None, regr_list = None, 
                batch_size=1):
    '''
    Returen a list of indices in X_pool that are selected randomly, 
    it often acts as a baseline method in Active learning.

    Keyword arguments:
    X_pool -- the potential experiment matrix
    batch_size -- the number of instances returned (default 1)
    '''

    #Random sampling without replacement
    Random_list = np.array(random.sample(range(len(X_pool)), batch_size))

    return Random_list

def Query_by_Committee(X_pool, X_train, 
                       y_train, regr_list = None, 
                       batch_size=1):
    '''
    Returen a list of indices in X_pool that are selected by Query-by-Committee method.
    QBC algorithm uses a committee to evaluate uncertainties of each instance in X_pool and return ones with highest uncertainty.

    Keyword arguments:
    X_pool -- the potential experiment matrix
    X_train -- the features of existing labelled dataset
    y_train -- the labels of existing labelled dataset
    regr_list -- a list of regressors that constructs a committee (default 4 GBDT regressors)
    batch_size -- the number of instances returned (default 1)
    '''
    # Instancing committee regressor prediction results
    committee_pred = np.zeros((len(regr_list),len(X_pool)))

    for i in range(len(regr_list)):
        # Bootstrapping from X_train data
        X_train_boot, y_train_boot = resample(X_train, y_train, random_state=i)
        regr_list[i].fit(X_train_boot, y_train_boot)

        # Predicting using trained committee from X in Pool
        committee_pred[i]=regr_list[i].predict(X_pool)
    
    # The ambiguity is measured by std in the predicted value
    pred = np.std(committee_pred, axis=0)

    # Return the argmax (batch_size) of std in the pred matrix
    if len(pred) > batch_size:
        QBC_list = pred.argsort()[-batch_size:][::-1]
    else:
        QBC_list = pred.argsort()[-len(pred) :][::-1]

    return QBC_list

def Greedy(X_pool, X_train = None, 
           y_train = None, regr_list = None, 
           batch_size=1):
    
    '''
    Returen a list of indices in X_pool that are selected by Greedy method.
    Greedy algorithm calculates and selects the samples that are the most distant from the original training set

    Keyword arguments:
    X_pool -- the potential experiment matrix
    X_train -- the features of existing labelled dataset
    batch_size -- the number of instances returned (default 1)
    '''
    
    # Instancing Euclidean distance matrix
    distance = np.zeros(len(X_pool))
    for i in range(len(X_pool)):
        distance[i] = np.sum(np.square(X_train - X_pool[i]))

    # Return the argmax (batch_size) of distance in the matrix
    if len(distance) > batch_size:
        Greedy_list = distance.argsort()[-batch_size:][::-1]
    else:
        Greedy_list = distance.argsort()[-len(distance) :][::-1]
    return Greedy_list
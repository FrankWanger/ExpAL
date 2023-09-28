import numpy as np
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor

class ExpAL:
    '''
    Core instance of active learning

    Args:
        eval_regr (BaseEstimator): The regressor to be used as the evaluator in active learning process
        batch_size (int): batch size of each active learning round (default 1)
        verbose (int): level of information displayed (default 0)

    Attributes:

        X_train -- The features of existing labelled dataset
        y_train -- The labels of existing labelled dataset
        X_pool -- The potential experiment matrix
        eval_regr: The regressor to be used as the evaluator in active learning process
        batch_size: batch size of each active learning round (default 1)
        ALround -- Current round of AL

    Methods:
        init(X, y, X_pool): Initialize the training data and pool.
        eval(**eval_kwargs): Evaluate the current dataset with the eval_regr using 'cross_validation' function in sklearn.
        query(query_method): Query the pool for new instances using a specified query method.
        add_results(X_new, y_new): Add newly labeled instances to the training set.
        remove_from_pool(query_list): Remove queried instances from the pool.
    '''
    def __init__(self,
                 eval_regr: BaseEstimator,
                 batch_size: int = 1,
                 verbose: int = 0,
                 ) -> None:
        '''
        Initialize the ActiveLearner.

        Args:
            eval_regr (BaseEstimator): The regressor to be used as the evaluator in active learning process
            batch_size (int): batch size of each active learning round (default 1)
        '''
        self.X_start = np.array([])
        self.y_start = np.array([])
        self.X_pool = np.array([])
        self.batch_size = batch_size
        self.eval_regr = eval_regr
        self.verbose = verbose
        self.ALround = 0

    def init(self,X_start,y_start,X_pool):
        '''
        Initialize training data and pool.

        Args:
        X_start (numpy.ndarray): Initial training data features.
        y_start (numpy.ndarray): Initial training data labels.
        X_pool (numpy.ndarray): The potential experiment matrix

        '''
        #Sanity check for X_pool and X_start
        assert X_pool.shape[1:] == X_start.shape[1:], 'Shapes has to match for X_pool and X except for the 1st dimention' 

        if len(self.X_start) == 0:
            self.X_start = X_start
            self.y_start = y_start
            self.X_pool = X_pool
            if self.verbose > 0:
                print('Shape of Starting Set: {}'.format(self.X_start.shape))
                print('Shape of Pool: {}'.format(self.X_pool.shape))
                print('***Initial data added***')
            
        else:
            raise Exception('Data already exists!')
        return 

    def eval(self, **eval_kwargs):
        '''
        Evaluate the regressor using cross-validation.

        Args:
        **eval_kwargs: Additional keyword arguments to be passed to cross_validate.

        Returns:
        dict: Cross-validation results.
        '''
        scoring = eval_kwargs.pop('scoring','neg_root_mean_squared_error')
        X_start,y_start = shuffle(self.X_start, self.y_start)
        return cross_val_score(self.eval_regr,X_start,y_start, scoring=scoring, **eval_kwargs)
        
    def query(self, query_method: Callable, **kwargs):
        '''
        Core step in active learning. Query the pool for new instances using a specified query method.
        Query by Committee by default uses 4 GBDT regressors as the committee.

        Args:
        query_method (Callable): The query method to use.
        **kwargs: Additional keyword arguments to be passed to query_method

        Returns:
        tuple: A tuple containing the queried instances (X_query) and their indices (query_idxs).
        '''
        regr_list = None

        #Specific settings for QBC method
        if query_method.__name__=='Query_by_Committee':
            regr_list = kwargs.pop('regr_list',[GradientBoostingRegressor() for i in range(4)])
            assert isinstance(regr_list,list), '\'regr_list\' has to be a list'
            assert all(isinstance(regr,BaseEstimator) for regr in regr_list), 'Elements in \'regr_list\' has to be BaseEstimator implemented in sklearn'

        #Perform active learning query
        query_idxs = query_method(X_pool=self.X_pool,
                                  X_train = self.X_start,
                                  y_train = self.y_start,
                                  batch_size=self.batch_size,
                                  regr_list=regr_list,
                                  **kwargs)
        
        #Retrieve X from the indices provided by query_method
        X_query = self.X_pool[query_idxs]

        # X_query = np.array([])
        # for idx in query_idxs:
        #     if X_query.size == 0:
        #         X_query = self.X_pool[idx][np.newaxis,:]
        #     else:
        #         X_query = np.vstack([X_query, self.X_pool[idx][np.newaxis,:]]) 
        return X_query,query_idxs
    
    def add_results(self,X_new,y_new):
        '''
        Add newly labeled instances to the training set.

        Args:
        X_new (numpy.ndarray): Newly labeled data features.
        y_new (numpy.ndarray): Corresponding labels.
        '''
        self.X_start = np.vstack([self.X_start, X_new]) 
        self.y_start = np.append(self.y_start, y_new)
        if self.verbose > 0: 
            print('Size in Starting Set: {}'.format(self.X_start.shape))

        #adding results to starting set completes one round of AL
        self.ALround = self.ALround + 1 

        return

    def remove_from_pool(self,query_idxs):
        '''
        Remove already queried instances from the pool.

        Args:
        query_idxs (list): List of indices to remove from the pool.
        '''
        self.X_pool = np.delete(self.X_pool, query_idxs, axis=0)
        if self.verbose > 0:
            print('Size in Pool: {}'.format(self.X_pool.shape))
        return 


class ExpALRetro(ExpAL):
    def __init__(self,
                 eval_regr: BaseEstimator,
                 batch_size: int = 1,
                 verbose: int = 0,
                 ) -> None:
        '''
        Initialize the ActiveLearner.

        Args:
            eval_regr (BaseEstimator): The regressor to be used as the evaluator in active learning process
            batch_size (int): batch size of each active learning round (default 1)
        '''
        super().__init__(eval_regr,batch_size,verbose)
        self.y_pool = np.array([])


    def initialize_data(self,X_start,y_start,X_pool,y_pool):
        '''
        Initialize training data and pool.

        Args:
        X_start (numpy.ndarray): Initial training data features.
        y_start (numpy.ndarray): Initial training data labels.
        X_pool (numpy.ndarray): The potential experiment matrix
        y_pool (numpy.ndarray): The labels for potential experiment matrix (only in retrospective AL)
        '''
        #Sanity check for X_pool and X_start
        assert X_pool.shape[1:] == X_start.shape[1:], 'Shapes has to match for X_pool and X except for the 1st dimention' 

        if len(self.X_start) == 0:
            self.X_start = X_start
            self.y_start = y_start
            self.X_pool = X_pool
            self.y_pool = y_pool
            if self.verbose > 0:
                print('Shape of Starting Set: {}'.format(self.X_start.shape))
                print('Shape of Pool: {}'.format(self.X_pool.shape))
                print('***Initial data added***')
            
        else:
            raise Exception('Data already exists!')
        return 
        
    def remove_from_pool(self,query_idxs):
        '''
        Remove already queried instances from the pool.

        Args:
        query_idxs (list): List of indices to remove from the pool.
        '''
        self.X_pool = np.delete(self.X_pool, query_idxs, axis=0)
        self.y_pool = np.delete(self.y_pool, query_idxs, axis=0)

        if self.verbose > 0:
            print('Size in Pool: {}'.format(self.X_pool.shape))
        return 
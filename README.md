# ExpAL: Active Learning Framework for Experimentalists

The `ExpAL` is a simplified active learning toolbox with a focus on tackling data scarcity issue in experimental studies. It fits well with the current machine learning framework sci-kit learn  and can be easily implemented by experimental scientists.

This code repository provides a versatile active learning framework for regression tasks. It includes two main classes: `ExpAL` and `ExpALRetro`, along with various sampling methods for selecting instances from a pool of unlabeled data.

## Installation

This code requires the following Python libraries to be installed:
- `numpy`
- `sklearn` (Scikit-learn)
- `pandas`

You can install ExpAL directly from source:

```
pip install git+https://github.com/FrankWanger/ExpAL.git
```

## Usage
The `ExpAL` class in this code repository provides a flexible framework for active learning, including various functionalities such as data initialization, evaluation, querying, and result management. It is designed to work with regression tasks in experiments

1. **Initialize the Active Learner:**
   ```python
   import numpy as np
   from sklearn.ensemble import GradientBoostingRegressor
   from ExpAL.core import ExpAL
   from ExpAL.methods import Random_Pick, Query_by_Committee, Greedy

   eval_regr = GradientBoostingRegressor()  # Create a regressor for modeling performance evaluation

   al = ExpAL(eval_regr=eval_regr, batch_size=1)
   ```

2. **Initialize with Data and evaluate the initial performance:**
   ```python
   al.init(X_start, y_start, X_pool)
   al.eval()
   ```

3. **Query for New Instances:**
   ```python
   X_query, query_idxs = al.query(Query_by_Committee)
   ```
4. **Perform some lab experiments to label X_query(not through the following function!)**
   ```python
   y_new = Some_Labor_Intensive_Experiments(X_new)
   ```
5. **Add New Instances to Training Data and remove queried ones from Pool:**
   ```python
   al.add_results(X_new, y_new)
   al.remove_from_pool(query_idxs)
   ```

6. **Evaluate the Model again to see improvements:**
   ```python
   al.eval()
   ```
7. **Repeat!**

### Methods
- `init(X_start, y_start, X_pool)`: Initialize training data and pool.
- `eval(**kwargs)`: Evaluate the regressor using cross-validation.
- `query(query_method, **kwargs)`: Query the pool for new instances using a specified query method.
- `add_results(X_new, y_new)`: Add newly labeled instances to the training set.
- `remove_from_pool(query_idxs)`: Remove already queried instances from the pool.

### The `ExpALRetro` Class
Additionally, this repository includes an extension of the `ExpAL` class called `ExpALRetro`. It provides the same functionality as `ExpAL` but with support for retrospective active learning, where labels for the potential experiment matrix are available.

### Available Query Methods

The code repository includes three query methods:

1. **Random_Pick**: Randomly selects instances from the pool as a baseline active learning strategy.

2. **Query_by_Committee**: Selects instances using the Query-by-Committee (QBC) method, which relies on a committee of regressors to evaluate uncertainty.

3. **Greedy**: Selects instances that are the most distant from the original training set based on Euclidean distance.




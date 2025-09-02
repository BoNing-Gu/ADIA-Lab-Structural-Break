# Install the Crunch CLI
#%pip install --upgrade crunch-cli

# Setup your local environment
#!crunch setup --notebook structural-break hello --token aaaabbbbccccddddeeeeffff


import os
import typing

# Import your dependencies
import joblib
import pandas as pd
import scipy
import sklearn.metrics


import crunch

# Load the Crunch Toolings
#crunch = crunch.load_notebook()


# Load the data simply
#X_train, y_train, X_test = crunch.load_data()


#X_train


#y_train


#print("Number of datasets:", len(X_test))


#X_test[0]


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    # For our baseline t-test approach, we don't need to train a model
    # This is essentially an unsupervised approach calculated at inference time
    model = None

    # You could enhance this by training an actual model, for example:
    # 1. Extract features from before/after segments of each time series
    # 2. Train a classifier using these features and y_train labels
    # 3. Save the trained model

    joblib.dump(model, os.path.join(model_directory_path, 'model.joblib'))


def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    model = joblib.load(os.path.join(model_directory_path, 'model.joblib'))

    yield  # Mark as ready

    # X_test can only be iterated once.
    # Before getting the next dataset, you must predict the current one.
    for dataset in X_test:
        # Baseline approach: Compute t-test between values before and after boundary point
        # The negative p-value is used as our score - smaller p-values (larger negative numbers)
        # indicate more evidence against the null hypothesis that distributions are the same,
        # suggesting a structural break
        def t_test(u: pd.DataFrame):
            return -scipy.stats.ttest_ind(
                u["value"][u["period"] == 0],  # Values before boundary point
                u["value"][u["period"] == 1],  # Values after boundary point
            ).pvalue

        prediction = t_test(dataset)
        yield prediction  # Send the prediction for the current dataset

        # Note: This baseline approach uses a t-test to compare the distributions
        # before and after the boundary point. A smaller p-value (larger negative number)
        # suggests stronger evidence that the distributions are different,
        # indicating a potential structural break.


#crunch.test(
#    # Uncomment to disable the train
#    # force_first_train=False,
#
#    # Uncomment to disable the determinism check
#    # no_determinism_check=True,
#)


#prediction = pd.read_parquet("data/prediction.parquet")
#prediction


# Load the targets
#target = pd.read_parquet("data/y_test.reduced.parquet")["structural_breakpoint"]

# Call the scoring function
#sklearn.metrics.roc_auc_score(
#    target,
#    prediction,
#)

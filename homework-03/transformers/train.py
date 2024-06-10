import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    categorical = ['PULocationID', 'DOLocationID']
    def train(df):
        dv = DictVectorizer()
        
        train_dicts = df[categorical].to_dict(orient='records')
        #Fit dict vectorizer
        X_train = dv.fit_transform(train_dicts)
        target = 'duration'
        y_train = df[target].values

        lr = LinearRegression()
        #train linear regression with default parameters
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        print(lr.intercept_)
        return lr, dv

    lr, dv = train(df)
    return lr, dv
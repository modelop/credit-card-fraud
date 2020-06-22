import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, auc, roc_curve


import pickle


# modelop.init
def begin():
    pass


# modelop.train
def train(traind_df):
    pass


# modelop.metrics
def metrics(df):
    yield {"tpr": .50, "fpr": .75}


# modelop.score
def predict(X):
    for row in X:
        yield 0






class GetDistance(BaseEstimator, TransformerMixin):
    # Class Constructor


    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

        # Custom transform method we wrote that creates aformentioned features and drops redundant ones

    def transform(self, X, y=None):
        # Check if needed
        X['distance'] = X['merchant_zipcode'] - X['customer_zipcode']
        X.drop(['merchant_zipcode', 'customer_zipcode'], axis=1, inplace=True)

        return X.values




def _begin():
    global model
    model = pickle.load(open('model.pkl', 'rb'))


def _train(train_df):

    X_train = train_df.drop('fraud_label', axis=1)
    y_train = train_df['fraud_label']
    X_train = X_train.drop(['transaction_id', 'card_id', 'customer_id', 'merchant_id', 'merchant_name'], axis=1)  #no predictive value in these fields


    numeric_transformer = Pipeline(steps=[
        ('distance', GetDistance()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = ['transaction_amount','merchant_zipcode','customer_zipcode']

    categorical_features = X_train.select_dtypes(include=['object']).columns


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])



    model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(max_depth=30))])

    model.fit(X_train, y_train)

    pickle.dump(model, open('model.pkl', 'wb'))



def _metrics(df):

    X_test = df.drop('fraud_label', axis=1)
    y_test = df['fraud_label']
    X_test = X_test.drop(['transaction_id', 'card_id', 'customer_id', 'merchant_id', 'merchant_name'],
                           axis=1)  # no predictive value in these fields

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_score = auc(fpr, tpr)
    yield { "ACCURACY": accuracy, "AUC": auc_score}



def _predict(X):
    df = pd.DataFrame(X, index=[0])
    y_pred = model.predict(df)
    for p in y_pred:
        yield p


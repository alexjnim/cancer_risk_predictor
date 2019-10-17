# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os

np.random.seed(42)

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd
pd.set_option('display.max_rows', 30)

df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# +
import seaborn as sns

def corr_matrix(df, attribute_list, key_attribute):
    new_df = pd.DataFrame()
    for i in attribute_list:
        new_df[i] = df[i]
            
    matrix = new_df.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu")
    
    print(matrix[key_attribute].sort_values(ascending=False))
    
    return 


# -

def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats



def _get_numerical_features(df, cat_list):
    feats = [col for col in list(df.columns) if col not in cat_list]
    return feats


cat_feats = _get_categorical_features(df)

num_feats = _get_numerical_features(df, cat_feats)

corr_matrix(df, num_feats, 'Biopsy')

# # changing the '?' in all the categorical features into the median or delete and changing certain categorical features into numerical features

cat_feats


def delete_q(df, l):
    for i in l:
        df = df[df[i] != '?']
        df[i].value_counts()
        print(i)
        print(df.shape)
    return df


def q_to_median(df, l):
    for i in l:
        df[i] = df[i].replace(['?'], [df[i].mode()])
        df[i] = df[i].astype(float)
        print(i)
        print(df.shape)
    return df


df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)

l = ['Number of sexual partners',
     'First sexual intercourse',
     'Num of pregnancies',
     'Smokes (years)',
     'Smokes (packs/year)', 
     'Hormonal Contraceptives (years)', 
     'IUD (years)',
     'STDs (number)']

l2 = [ 'Smokes',
 'Hormonal Contraceptives',
 'IUD',
 'STDs',
 'STDs:condylomatosis',
 'STDs:cervical condylomatosis',
 'STDs:vaginal condylomatosis',
 'STDs:vulvo-perineal condylomatosis',
 'STDs:syphilis',
 'STDs:pelvic inflammatory disease',
 'STDs:genital herpes',
 'STDs:molluscum contagiosum',
 'STDs:AIDS',
 'STDs:HIV',
 'STDs:Hepatitis B',
 'STDs:HPV',]





df = delete_q(df, l2)

df = q_to_median(df,l)

cat_feats = _get_categorical_features(df)
num_feats = _get_numerical_features(df, cat_feats)

cat_feats

num_feats

corr_matrix(df, num_feats, 'Biopsy')

X_train = df.drop('Biopsy', axis = 1)
cat_feats = _get_categorical_features(X_train)
num_feats = _get_numerical_features(X_train, cat_feats)

y_train = df['Biopsy']



# # now let's put the data through a pipeline

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class selector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
      


num_pipeline = Pipeline([
            ('selector', selector(num_feats)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
                    ])

cat_pipeline = Pipeline([
                ('selector', selector(cat_feats)),
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('cat_encoder', OneHotEncoder(sparse=False)),
])

# +
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
# -

X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)


X_train_processed



X_train.to_csv(r'X_train.csv')
y_train.to_csv(r'y_train.csv')

X_train_processed.to_csv(r'X_train_processed.csv')

y_train.shape

y_train.to_csv(r'y_train2.csv')



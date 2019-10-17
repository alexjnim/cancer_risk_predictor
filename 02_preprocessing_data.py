# -*- coding: utf-8 -*-
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
# -

df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)

df.shape

df.head()

t = df[df != '?'] #this will change all ? into nan

t.head()

# seperate the dataframe into one with all rows that have nan values, and the other with no nan values
nan_df = t[t.isna().any(axis=1)]
nonan_df = t.dropna()

nonan_df.head()

# +
from sklearn.model_selection import train_test_split

# here we make a X_pretest set that has 2/10 of the rows that don't have nan values. that is 2/10 * 77271 = 15450, 
# so 15% of the training dataset

X_test, remaining_nonan = train_test_split(nonan_df, train_size= 0.2, random_state = 42)
# -

X_test.isnull().any(axis=0).sum()

df = pd.concat([nan_df, remaining_nonan])

df.shape


def print_totshape(df1, df2):
    m,n = df1.shape
    a,b = df2.shape
    
    print(m+a)
    return


print_totshape(df, X_test)

# # quick look at the correlation matrix

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

# # filling the nan values and changing certain categorical features into numerical features

cat_feats


def fill_mode(df, attribute_list):
    for i in attribute_list:
        print(i)
        print(df[i].dtype)
        df[i].fillna(df[i].mode()[0], inplace=True)
        df[i] = df[i].astype(float)
        print(df[i].dtype)
    return df


def convert_num(df, attribute_list):
    for i in attribute_list:
        print(i)
        print(df[i].dtype)
        df[i] = df[i].astype(float)
        print(df[i].dtype)
    return df


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

df = fill_mode(df, l)

X_test = convert_num(X_test, l)

df = df.dropna(how='any',axis=0) 

cat_feats = _get_categorical_features(df)
num_feats = _get_numerical_features(df, cat_feats)

cat_feats

num_feats

corr_matrix(df, num_feats, 'Biopsy')

# # creating X and y data

X_train = df.drop('Biopsy', axis = 1)
cat_feats = _get_categorical_features(X_train)
num_feats = _get_numerical_features(X_train, cat_feats)

y_train = df['Biopsy']
y_test = X_test['Biopsy']
X_test = X_test.drop('Biopsy', axis = 1)

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

# +
X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)

# note that we are using transform() on X_pretest 
#we standardize our training dataset, we need to keep the parameters (mean and standard deviation for each feature). 
#Then, we use these parameters to transform our test data and any future data later on
#fit() just calculates the parameters (e.g. 𝜇 and 𝜎 in case of StandardScaler) 
#and saves them as an internal objects state. 
#Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
X_test_processed = full_pipeline.transform(X_test)
X_test_processed = pd.DataFrame(X_test_processed)



# -


X_train.to_csv(r'X_train.csv')
X_test.to_csv(r'X_test.csv')
y_train.to_csv(r'y_train.csv')
y_test.to_csv(r'y_test.csv')


X_train_processed.to_csv(r'X_train_processed.csv')
X_test_processed.to_csv(r'X_test_processed.csv')

y_test.value_counts()

y_train.value_counts()



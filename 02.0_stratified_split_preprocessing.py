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

# + {"endofcell": "--"}
import numpy as np
import os

np.random.seed(42)

# # %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# # +
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

# --

# +
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)

for train_index, test_index in split.split(nonan_df, nonan_df["Biopsy"]):
    strat_train_set = nonan_df.loc[train_index] 
    strat_test_set = nonan_df.loc[test_index]
# -

strat_train_set['Biopsy'].value_counts()

strat_test_set['Biopsy'].value_counts()

df = pd.concat([nan_df, strat_train_set])


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


# +

def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats



def _get_numerical_features(df, cat_list):
    feats = [col for col in list(df.columns) if col not in cat_list]
    return feats


cat_feats = _get_categorical_features(df)

num_feats = _get_numerical_features(df, cat_feats)
# -

corr_matrix(df, list(df.columns), 'Biopsy')


# +

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

strat_test_set = convert_num(strat_test_set, l)

df = df.dropna(how='any',axis=0) 
# -

corr_matrix(df, df.columns, 'Biopsy')





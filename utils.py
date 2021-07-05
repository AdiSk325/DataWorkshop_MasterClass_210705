import pandas as pd
import numpy as np
import helper as h

from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error as mae
import eli5
from tqdm import tqdm

import gc

def get_list_el(lst, idx):
    "Funkcja do rozbicia list na poszczególne kolumny"
    try: 
        return lst[idx]
    
    except: 
        return None    
    
    
def convert_dict_to_df_cols(params_dict, index_name=0):
    return pd.DataFrame(index=params_dict.keys(), data=params_dict.values(), columns=[index_name]).T


#Funkcje przekształcające zmienną celu "orig_price_trans"
def id_func(*kwargs): 
    "Funkcja identycznościowa dla check_model"
    return kwargs[0]


def price_factr_area_trans(y_pred, X_cv_test): 
    """Cena za m2 * powierzchnia mieszkania"""
    return y_pred*X_cv_test['Общая площадь:']


def log_price_factr_area_trans(y_pred, X_cv_test): 
    """Cena za m2 * powierzchnia mieszkania"""
    return np.exp(y_pred)*X_cv_test['Общая площадь:']


def check_model(df, feats, price,  model, orig_price_trans=id_func, n_splits=5, scoring="neg_mean_absolute_error"):
    df_train = df[~df[price].isna()].copy()
    
    X_train = df_train[feats].reset_index(drop=True)
    y_train = df_train[price].reset_index(drop=True)
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    
    for train_idx, test_idx in tqdm(cv.split(X_train)):
        
        X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_cv_train, y_cv_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_cv_train, y_cv_train)
        y_pred = model.predict(X_cv_test)
        
        y_pred = orig_price_trans(y_pred, X_cv_test)
        y_test = orig_price_trans(y_cv_test, X_cv_test)
        
        score = mae(y_test, y_pred)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)











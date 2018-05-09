import os
import gc
import csv
import sys
import ast
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score

from imblearn.over_sampling import SMOTE
from sklearn.exceptions import DataConversionWarning

csv.field_size_limit(sys.maxsize)
os.chdir('../Data/Seattle/')

def sublist_uniques(df,sublist):
    '''
    https://stackoverflow.com/questions/36487842/python-pandas-how-to-create-a-binary-matrix-from-column-of-lists
    '''
    categories = set()
    for d,t in df.iterrows():
        try:
            for j in ast.literal_eval(t[sublist]):
                categories.add(j)
        except:
            pass
    return list(categories)

def sublists_to_dummies(df,sublist,index_key = None):
    '''
    Create a binary matrix from column of lists
    Sample usage: sublists_to_dummies(df,'cuisines')

    '''
    categories = sublist_uniques(df,sublist)
    frame = pd.DataFrame(columns=categories)
    for d,i in df.iterrows():
        sub_lst = ast.literal_eval(i[sublist])
        if type(sub_lst) == list or np.array:
            try:
                if index_key != None:
                    key = i[index_key]
                    f =np.zeros(len(categories))
                    for j in sub_lst:
                        f[categories.index(j)] = 1
                    if key in frame.index:
                        for j in sub_lst:
                            frame.loc[key][j]+=1
                    else:
                        frame.loc[key]=f
                else:
                    f =np.zeros(len(categories))
                    for j in sub_lst:
                        f[categories.index(j)] = 1
                    frame.loc[d]=f
            except:
                pass

    return frame


# Basic Data Cleaning 
def data_clean(sample_frac):
    df = pd.read_csv('seattle_instances_mergerd.csv', sep=None,engine='python')
    df = df_output.sample(frac=sample_frac)

    # Convert to datetime object
    df.inspection_period_start_date = pd.to_datetime(df.inspection_period_start_date)
    df.inspection_period_end_date = pd.to_datetime(df.inspection_period_end_date)

    df['inspection_year'] = df["inspection_period_end_date"].dt.year
    df['inspection_month'] = df["inspection_period_end_date"].dt.month

    # Remove inpection_score = -1 outlier
    df = df[df.inspection_penalty_score >= 0]

    # Create a prediction label with threshold of 40
    df['label'] = np.where(df['inspection_penalty_score']>=50, 1, 0)

    # Create a binary variable from column of lists
    df = sublists_to_dummies(df,'cuisines')
        
    return df




df1 = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1) # for stage 2 ML
df2 = df[['content','label']] # for stage 2 NLP ML
 

# Stage 2 ML Model Pipilines
models = {'LR': LogisticRegression(),
		'NB': GaussianNB()}

params = {'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
		  'NB': {}}

def ml_pipline(df,models,params):
	y = df.label
	X = df.iloc[:,:-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	X_train_res, y_train_res = SMOTE().fit_sample(X_train, y_train)

	for key, model in models.items():
		param = params[key]
		clf = GridSearchCV(model, param)
		pred = clf.fit(X_train_res, y_train_res).predict(X_test) 
		print('-------Model--'+key+"-------")
		print(clf.best_estimator_)
		print(classification_report(y_test, pred))
		print(accuracy_score(y_test, pred))

# Test 
ml_pipline(df1,models,params)   


# NLP Machine learning 



   



                                     






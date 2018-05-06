import os
import gc
import csv
import sys
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score

from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
csv.field_size_limit(sys.maxsize)
os.chdir('../Data/Seattle/')


# Basic Data Cleaning 
df_output = pd.read_csv('review_topic_model_seattle.csv', sep=None,engine='python')
df = df_output.sample(frac=0.3)
# Create a prediction label with threshold of 40
df['label'] = np.where(df['inspection_penalty_score']>=40, 1, 0)
df1 = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1) # for stage 2 ML
df2 = df[['content','label']]


# Stage 2 ML Model Pipilines
models = {'LR': LogisticRegression(),
		'NB': GaussianNB()}

params = {'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
		  'NB': {}}

def ml_pipline(df,models,params):
	y = df.label
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
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



   



                                     






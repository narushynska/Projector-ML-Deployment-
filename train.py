import os
import pickle

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from NLP_preprocessor import NltkTextPreprocessor

print('read dataset')
X = pd.read_csv("data/train.csv")

print('Split dataset into train and test')
X_train, X_test, y_train, y_test = train_test_split(X['excerpt'], X['target'], test_size=0.2, random_state=11)

print('Transform data and train the model')
prediction_pipeline = Pipeline(steps=[
           ('text_preproc', NltkTextPreprocessor()), 
           ('tfidf', TfidfVectorizer()),
           ('RFRegressor', RandomForestRegressor(random_state=11))])
prediction_pipeline.fit(X_train,y_train)

y_pred = prediction_pipeline.predict(X_test)
print('RMSE for Random forest regressor is ',r2_score(y_test, y_pred))

if not os.path.exists('model'):
   os.makedirs('model')
pickle.dump(prediction_pipeline,open('model/pipeline.pickle','wb'))
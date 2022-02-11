import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import r2_score
df = pd.read_csv('Concrete_Data.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
RANDOM_STATE = 24
np.random.seed(RANDOM_STATE)
Kfold = KFold(n_splits=5,random_state=RANDOM_STATE,shuffle=True)
params = {'n_estimators':np.arange(50,200,50),'max_depth':np.arange(2,30,1)}
rf = RandomForestRegressor(random_state=RANDOM_STATE)
cv = GridSearchCV(rf, param_grid=params,verbose=3,cv=Kfold,scoring='r2')
cv.fit(x,y)
cv.best_score_
cv.best_estimator_
cv.best_params_
df_test = pd.read_csv("testConcrete.csv")
model = cv.best_estimator_
pred=model.predict(df_test)
pred
df_test['Strength_prediction']=pred
df_test.to_csv('strength_predictions.csv',index=False)

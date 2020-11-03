import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("df_clean.csv")

# Choose relevant features
df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'hourly', 'employer_provided',
             'job_state', 'same_state', 'age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split
X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)
print(np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# lasso regression
lm_1 = Lasso()
print(np.mean(cross_val_score(lm_1, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

alpha = []
error = []
for i in range(1, 100):
    alpha.append(i/10)
    lml = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lm_1, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

#plt.plot(alpha, error)
#plt.show()

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print(np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# Tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 300, 10), 'criterion':('mse', 'mae'), 'max_features':('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_estimator_)

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_rf))
print(mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2))

# save model
import pickle
pickl = {'model': gs.best_estimator_, 'test': list(X_test.iloc[1,:])}
pickle.dump(pickl, open('model_file' + ".p", "wb"))
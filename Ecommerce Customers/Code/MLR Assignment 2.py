# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:18:15 2019

@author: Abhishek Jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

data = pd.read_csv('Ecommerce Customers.csv')
data.head()

X = data.iloc[:,3:-1]
y = data.iloc[:,-1]

sns.heatmap(data.isnull(),yticklabels=False,cbar = False,cmap='viridis' )

correlation = data.corr()


# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor,X = X_train,y = y_train, cv=10)
print(accuracies.mean())
print(accuracies.std()) # less std means the result is not all over the place

def r_squared(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r^2 = {:.3f}".format(r**2),
                xy=(.1, .9), xycoords=ax.transAxes)

# Check for Linearity visually
linearity_assumption_plot_1 = sns.pairplot(pd.DataFrame(X), kind="reg")
linearity_assumption_plot_1.map_lower(r_squared)

error_residual = pd.DataFrame(y_test-y_pred)
error_residual.reset_index(inplace = True)
linearity_test_df = pd.DataFrame(X_test)
linearity_test_df['Residual'] = error_residual['Yearly Amount Spent']
linearity_test_df.columns = 'Avg. Session Length#Time on App#Time on Website#Length of Membership#Residuals'.split('#')

linearity_assumption_plot_2 = sns.pairplot(linearity_test_df.iloc[:, 2:], kind="reg")
linearity_assumption_plot_2.map_lower(r_squared)

endogenity_check = linearity_test_df.corr() # Check only the reciduals row with other data
#endogenity_check

residual_test = np.column_stack([y_test,y_pred])
residual_test = pd.DataFrame(residual_test)
residual_test.columns='Y_test predictions'.split()
sns.jointplot(x='Y_test', y='predictions', data=residual_test, kind='reg')

stats.levene(residual_test['Y_test'], residual_test['predictions']) # check p value > threshold(0.05), levene test for homoscedecity passed 

stats.shapiro(error_residual['Yearly Amount Spent'])

#the coefficients
print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error, r2_score

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
print("Variance score: {}".format(r2_score(y_test, y_pred)))

# Building the optimal model using Backward Elimination
# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(X)


x = pd.DataFrame(x, columns = 'Avg. Session Length#Time on App#Time on Website#Length of Membership#Residuals'.split('#'))


X_opt = x.loc[:, ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
#X_opt
# Fit the model, according to the OLS (ordinary least squares)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = x.loc[:, ['Avg. Session Length', 'Time on App', 'Length of Membership']]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

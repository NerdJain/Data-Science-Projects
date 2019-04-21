# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:43:24 2019

@author: Abhishek Jain
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')






'''
Logistic regression does not make many of the key assumptions of linear
 regression and general linear models that are based on ordinary least squares
 algorithms – particularly regarding linearity, normality, homoscedasticity,
 and measurement level.
 
First, logistic regression does not require a linear relationship between the
 dependent and independent variables.  
Second, the error terms (residuals)  do not need to be normally distributed.
Third, homoscedasticity is not  required.  Finally, the dependent variable 
in logistic regression is not measured on an interval or ratio scale. 
'''







'''
Binary logistic regression requires the dependent variable to be binary
 and ordinal logistic regression requires the dependent variable to be ordinal.

Logistic regression requires the observations to be independent of each
 other.  In other words, the observations should not come from repeated
 measurements or matched data.
 
Logistic regression typically requires a large sample size. 
 A general guideline is that you need at minimum of 10 cases with the least 
 frequent outcome for each independent variable in your model. For example, 
 if you have 5 independent variables and the expected probability of your 
 least frequent outcome is .10, then you would need a minimum sample 
 size of 500 (10*5 / .10).
'''






'''# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])'''
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')






X = dataset.iloc[:, [2, 3]]
y = dataset.iloc[:, -1]






'''No multicolinearity - also check for condition number
Logistic regression requires there to be little or no multicollinearity
 among the independent variables.  This means that the independent variables
 should not be too highly correlated with each other.
 
We observe it when two or more variables have a high coorelation.
If a can be represented using b, there is no point using both
c and d have a correlation of 90% (imprefect multicolinearity). if c can be almost
represented using d there is no point using both
FIX : a) Drop one of the two variables. b) Transform them into one variable by taking
mean. c) Keep them both but use caution. 
Test : before creating the model find correlation between each pairs.
'''
dataset.corr()



'''
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 3] = labelencoder_X.fit_transform(X.iloc[:, 3])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''





# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)





# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)






# Predicting the Test set results
y_pred = classifier.predict(X_test)






# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()






# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)




import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))





#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



# Only for learning, 2 feature data
# Visualising the Training set results
def visualise_logestic_regression_results(X_set, y_set, xlabel, ylabel):
    from matplotlib.colors import ListedColormap
    
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
# Visualising the Training set results
visualise_logestic_regression_results(X_train, y_train, 'Age', 'Estimated Salary')

# Visualising the Test set results
visualise_logestic_regression_results(X_test, y_test, 'Age', 'Estimated Salary')
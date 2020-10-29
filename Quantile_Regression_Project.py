"""
@authors: Yani Bouaffad, Ryma Lakehal and Loïc Sauton
In short: Quantile regression using CVXCP package.
"""

# -*- coding: utf-8 -*-


################################
# Packages needded
#################################

import numpy as np
import pandas as pd
import cvxpy as cvx
from download import download
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import time
import seaborn as sns
from sklearn.linear_model import Ridge
from numpy.lib.function_base import quantile
import random
import statsmodels.formula.api as smf
import statsmodels.api as sm 
import scipy.stats as stat

################################
# Download datasets
#################################

## we were inspired by Joly Julien , Mohamed Sahardid et Anas El Benna's code 
## to import the dataset and use it later to compare thiers results with ours.
url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=True)
occ_raw = pd.read_csv(path_target)
occ_raw['date'] = pd.to_datetime(occ_raw['date_debut']).dt.to_period('M')


data = pd.read_csv(path_target)
data['date'] = pd.to_datetime(data['date_debut']).dt.to_period('M')

## Cities Data Visualisation 
plt.figure()
occ_raw["nom_com"].value_counts().plot.pie(autopct="%.1f%%")
plt.title("Cities Pie plot.")
plt.show()


## Pollulants Data Visualisation
sns.countplot(occ_raw['polluant'])
plt.title("pollutants Bar chart")

## We isolate the cities we are interested in a list "ville" and thiers Ozone readings from July 2018
ville = 'MONTPELLIER', 'TOULOUSE', 'AGDE'
df = occ_raw[occ_raw['nom_com'].isin(ville)]
df = df[df['polluant'] == 'O3']


df.date.unique()
df = df[df.date == '2018-07']
df = df[['nom_com', 'valeur_originale']]

## Boxplot of the chosen cities
sns.boxplot('nom_com', 'valeur_originale', data=df)
plt.title("Boxplot of the chosen Cities")


## the cities recoded into qualitative variables
valeur_o3 = []
len_villes = []

for i in range(len(ville)):

    o3_i = df[df['nom_com'] == ville[i]].values[:, 1]

    valeur_o3 = np.append(valeur_o3, o3_i)

    len_villes = np.append(len_villes, o3_i.shape)

X = np.zeros((int(np.sum(len_villes)), len(ville)))

for i in range(len(ville)):
    X[int(np.sum(len_villes[0:i])):int(np.sum(len_villes[0:i + 1])
                                       ), i] = np.ones(
                                           int(np.sum(len_villes[i])))



################################
# Quantile regression
#################################

### Implementing quantile regression using CVXPY package
Y = valeur_o3
X = X

b = cvx.Variable(len(X[1]))
alpha = cvx.Parameter(value=0.5)

error = 0
for i in range(len(X)):
    residuals = Y[i] - b.T * X[i]
    error += 0.5 * cvx.abs(residuals) + (alpha - 0.5) * residuals

objective = cvx.Minimize(error)
problem = cvx.Problem(objective)


## Solve quantile regression for 𝛼 = 0.5
fits = np.zeros(len(X))

start_time= time.time()  
problem.solve(solver="ECOS")
    
for i in range(len(X)):
    fits[i] = (X[i].T*b ).value

interval = time.time() - start_time
print("Prediction results with cvxpy package : ",np.unique(fits))

###############################################
# Implementation with statsmodels package
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start = time.time()
results = smf.quantreg('valeur_originale ~ nom_com', df)
res = results.fit(q=.5)
end = time.time()

prediction= (np.unique(res.predict(df[['nom_com']])))
print("Prediction results with package statsmodels : ",prediction)

################################
# Execution time Comparison
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("the execution time with the implementation obtained with cvxpy package : ",interval)
print("the execution time with the implementation obtained with package statsmodels : ",(end - start))

# the implementation with the statsmodels package is about 15 to 20 times better than 
# the implementation with the CVXPY package.

################################
# COMPARISON WITH RIDGE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###  Evolution of the Ridge estimator graphically

def iteration_ridge_alpha(logalpha):
    np.random.seed(0)
    current_palette = sns.color_palette()
    fig = sns.catplot(y="valeur_originale", x="nom_com", data=df,
                      jitter='0.05', legend_out=False, order=list(ville),
                      height=8)

    plt.axhline(y=np.mean(valeur_o3), xmin=0, xmax=len(ville), ls='--',
                linewidth=3, color='grey')
    plt.text(len(ville)-0.3, np.mean(valeur_o3), '$\\bar y$', color='grey',
             verticalalignment='bottom', horizontalalignment='right',
             fontsize=25)

    for i in range(len(ville)):

        y_predict = np.zeros(len(ville))
        y_predict[i] = 1
        model_ridge = Ridge(np.exp(logalpha))
        model_ridge.fit(X, valeur_o3)
        ridge = model_ridge.predict([y_predict])

        plt.scatter(i, np.mean(valeur_o3[int(np.sum(len_villes[0:i])):
                    int(np.sum(len_villes[0:i+1]))]), marker='_', lw=10,
                    color=current_palette[i], s=600)

        plt.text(i + 0.2, np.mean(valeur_o3[int(np.sum(len_villes[0:i])):
                 int(np.sum(len_villes[0:i+1]))]), '$\\bar y _{}$'.format(i),
                 verticalalignment='bottom', horizontalalignment='left',
                 color=current_palette[i], fontsize=20)

        plt.scatter(i, ridge, s=400, marker='_',
                    color=current_palette[len(ville)+i], lw=5,
                    label='ridge_{}'.format(ville[i]))

    plt.legend(loc='upper left', bbox_to_anchor=(1., .9))
    plt.tight_layout()
    plt.title('Evolution de l estimateur Ridge suivant la valeur \n'
              'du crit�re de p�nalisation alpha')


import ipywidgets as ip
ip.interact(iteration_ridge_alpha, logalpha=(-10, 10, 0.1))

## Nous avons fais une r�gression m�diane sur variables cat�gorielles

# On remarque que les estimations de Rigde se trouvent toutes autour de la moyenne.
# Pour Lambda �gale � z�ro l'estimateur OLS donne la moyenne des observation.
# Notre r�gression m�diane donne la m�diane des observations.
# Cela explique des r�sultats presque similaires.




###########################################################
# Reproduction of Figures 1.6 and 1.7 of Koenker's book
###########################################################


# set up

n = 60
sigma= 1
X = np.linspace(1,10, n)
epsilon = sigma*np.random.randn(n)
y = X + epsilon

data = pd.DataFrame( {"target": y , "features" : X})


quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

# Fig 1.6 for Koenker's Book

fig , ax = plt.subplots(1,5, sharey=True)
plt.suptitle("Quantile Regression for 5 quantiles")

for idx, q in enumerate(quantiles):
    
    model = smf.quantreg('target ~ features', data).fit(q)
    reg_line = model.params['Intercept'] + model.params['features'] * X 
    ax[idx].scatter(X, y, label="data")
    ax[idx].plot( X, reg_line, label= "q =" + str(q), color= "black")

true_line = [stat.norm.ppf(q, loc=0, scale= sigma^2) + X for q in quantiles]

for i in range(5):
    dic = {0: 'True line for 5 quantiles'}
    for j in range(5):
        ax[i].plot(X, true_line[j], label=dic[0], color='red', alpha = 0.5)
        dic[0] = "__noname__"
    ax[i].legend()
plt.show()

# Fig 1.7 for Koenker's Book
X_2 = X*X
eps_2 = np.random.randn(n)
y2 = 1+X +X_2 +eps_2

plt.plot(X,y2,'ro')
plt.show()

data2 = pd.DataFrame( {"target": y2 , "features" : X, 'features_squared': X_2})


fig , ax = plt.subplots(1,5, sharey=True)
plt.suptitle("Quantile Regression for 5 quantiles")

for idx, q in enumerate(quantiles):
    
    model2 = smf.quantreg('target ~ features + features_squared', data2).fit(q)
    reg_line2 = model2.params['Intercept'] + model2.params['features'] * X + model2.params['features_squared'] * X_2
    ax[idx].scatter(X, y2, label="data")
    ax[idx].plot( X, reg_line2, label= "q =" + str(q), color= "black")

true_line2 = [stat.norm.ppf(q, loc=0, scale= sigma**2) * X_2 + X +1 for q in quantiles]

for i in range(5):
    dic = {0: 'True line for 5 quantiles'}
    for j in range(5):
        ax[i].plot(X, true_line2[j], label=dic[0], color='red', alpha = 0.5)
        dic[0] = "__noname__"
    ax[i].legend()
plt.show()

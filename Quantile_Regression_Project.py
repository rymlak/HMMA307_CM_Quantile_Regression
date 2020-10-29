"""
@authors: Yani Bouaffad, Ryma Lakehal and LoÃ¯c Sauton
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

occ = occ_raw.dropna()

## Cities Data Visualisation 
plt.figure()
occ["nom_com"].value_counts().plot.pie(autopct="%.1f%%")
plt.title("Cities Pie plot.")
plt.show()


## Pollulants Data Visualisation
sns.countplot(occ['polluant'])
plt.title("pollutants Bar chart")

## We isolate the cities we are interested in a list "ville" and thiers Ozone readings from July 2018
ville = 'MONTPELLIER', 'TOULOUSE', 'AGDE'
df = occ[occ['nom_com'].isin(ville)]
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


### Solve quantile regression for different values of  ğ¼
# ğ¼={0.01,0.1,â¦,0.9,0.99}


## Solve quantile regression for 𝛼 = 0.5
fits = np.zeros(len(X))

start_time= time.time()  

problem.solve(solver="ECOS")

for k,a in enumerate(alpha):
    alpha.value = a
    problem.solve(solver="ECOS")

    
for i in range(len(X)):
    fits[i] = (X[i].T*b ).value


interval = time.time() - start_time
print("Median prediction obtained with cvxpy package : ",np.unique(fits))


interval= time.time() - start_time
print(np.unique(fits, axis=1))
print("the execution time with the implementation obtained with package cvxpy : ",interval)

################################
# Implementation with statsmodels package
#################################

start = time.time()
results = smf.quantreg('valeur_originale ~ nom_com', df)
res = results.fit(q=.5)
end = time.time()

prediction= (np.unique(res.predict(df[['nom_com']])))
print("Median prediction obtained with package statsmodels : ",prediction)

################################
# Execution time Comparison
#################################
print("the execution time with the implementation obtained with cvxpy package : ",interval)
print("the execution time with the implementation obtained with package statsmodels : ",(end - start))

print(prediction)


################################
# COMPARISON WITH RIDGE
#################################

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
              'du critère de pénalisation alpha')


import ipywidgets as ip
ip.interact(iteration_ridge_alpha, logalpha=(-10, 10, 0.1))

## Nous avons fais une régression médiane sur variables catégorielles
## La Régression Ridge est quant à elle sur variables continues

# On remarque que les estimations de Rigde se trouvent toutes autour de la moyenne.
# Pour Lambda égale à zéro l'estimateur OLS donne la moyenne des observation. Notre régression médiane 
# donne la médiane des observations.
# Cela explique des résultats presque similaires.

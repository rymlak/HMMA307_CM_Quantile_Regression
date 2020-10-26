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
import statsmodels.formula.api as smf
import time

################################
# Download datasets
#################################
url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=True)


data = pd.read_csv(path_target)
data['date'] = pd.to_datetime(occ['date_debut']).dt.to_period('M')


ville = 'MONTPELLIER', 'TOULOUSE', 'AGDE'
drm = occ[occ['nom_com'].isin(ville)]
drm = df[df['polluant'] == 'O3']


drm.date.unique()
drm = drm[drm.date == '2018-07']
drm = drm[['nom_com', 'valeur_originale']]

### the cities recoded into qualitative variables


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
alpha = cvx.Parameter()

for i in range(len(X)):
    residuals = Y[i] - b.T * X[i]
    error += 0.5 * cvx.abs(residuals) + (alpha - 0.5) * residuals

objective = cvx.Minimize(error)
problem = cvx.Problem(objective)

### Solve quantile regression for different values of  𝛼
# 𝛼={0.01,0.1,…,0.9,0.99}

fits = np.zeros((len(alphas), len(X)))
residuals_value = np.zeros((len(alphas), len(X)))
alphas = np.linspace(0.0, 1, 11)
alphas = np.r_[0.01, alphas[1:-1], 0.99]

start_time_1 = time.time()  
for k,a in enumerate(alphas):
    alpha.value = a
    problem.solve(solver="ECOS")
    
    for i in range(len(X)):
        fits[k,i] = (X[i].T*b ).value
        residuals_value[k,i] = (Y[i] - (X[i].T*b)).value

interval_1 = time.time() - start_time_1
np.unique(fits, axis=1)

################################
# Implementation with statsmodels package
#################################

start = time.time()
results = smf.quantreg('valeur_originale ~ nom_com', drm)
res = results.fit(q=.5)
end = time.time()
print("the execution time with the implementation obtained with package statsmodels : ",(end - start))

prediction= (np.unique(res.predict(df[['nom_com']])))
print(prediction)
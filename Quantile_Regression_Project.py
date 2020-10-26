# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


from download import download
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



start = time.time()
results = smf.quantreg('valeur_originale ~ nom_com', drm)
res = results.fit(q=.5)
end = time.time()
print("the execution time with the implementation obtained with package statsmodels : ",(end - start))

prediction= (np.unique(res.predict(df[['nom_com']])))
print(prediction)
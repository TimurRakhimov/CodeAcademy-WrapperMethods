# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
%matplotlib inline

# https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

# Load the data
obesity = pd.read_csv("obesity.csv")

# Inspect the data
print(obesity.head())

X = obesity.drop(['NObeyesdad'],axis=1)
y = obesity.NObeyesdad

lr = LogisticRegression(max_iter=1000)

lr.fit(X,y)

print(lr.score(X,y))

sfs = SFS(lr, forward=True, floating=False, k_features=9, scoring='accuracy', cv=0)

sfs.fit(X,y)

print(sfs.subsets_[9]['feature_names'], sfs.subsets_[9]['avg_score'])

plot_sfs(sfs.get_metric_dict())
plt.show()

sbs = SFS(lr, forward=False, floating=False, k_features=7, scoring='accuracy', cv=0)

sbs.fit(X,y)

print(sbs.subsets_[7])

print(sbs.subsets_[7]['feature_names'], sbs.subsets_[7]['avg_score'])

plt.clf()
plot_sfs(sbs.get_metric_dict())
plt.show()

features = X.columns

scaler = StandardScaler()
X = pd.DataFrame(StandardScaler().fit_transform(X))

rfe = RFE(lr, n_features_to_select=8)

rfe.fit(X,y)

rfe_features = [f for (f,support) in zip(features, rfe.support_) if support]
print(rfe_features)

print(rfe.score(X,y))

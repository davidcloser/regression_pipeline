import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

label = "label"
data = pd.read_csv("data/data.csv")

# Split the scores
data[label] = data[label].shift(-1)
data.dropna(inplace=True)
X = data.drop(label, axis=1)
y = data[label]

# Scale X
X = StandardScaler().fit_transform(X)

# Visualizing the correlation
sns.heatmap(data=data.corr(), annot=True)
plt.title("Correlation between the features")
plt.savefig("output/correlation_matrix.png")
plt.show()

# Get the most correlated columns
correlation = SelectKBest(score_func=f_regression, k="all").fit(X, y)
correlation_df = pd.DataFrame()
correlation_df["columns"] = data.drop(label, axis=1).columns.values
correlation_df["scores"] = correlation.scores_
correlation_df["pvalues"] = correlation.pvalues_
correlation_df.sort_values(by="scores", ascending=False, inplace=True)
correlation_df.to_csv("output/correlation_df.csv", index=False, sep=";")


# Define the model Dictionaries
alphas =  np.arange(0.2, 5, 0.1)
models = {
    "Lasso": (Lasso, {"alpha": alphas}),
    "AdaBoost": (AdaBoostRegressor, {"learning_rate": alphas, "loss": ["linear", "square", "exponential"]}),
    "RandomForest": (RandomForestRegressor, {"n_estimators": np.arange(10, 500, 10,)}),
    "SVR": (SVR, {"kernel": ["linear", "rbf", "sigmoid"]})
}


r2_scores = {}
for model_name, model in models.items():
    regressor = model[0]
    params = model[1]
    reg = GridSearchCV(regressor(), params, cv=3, n_jobs=-1, scoring="r2").fit(X, y)
    r2_scores[model_name] = {"r2": reg.best_score_, "params": reg.best_params_}

all_scores = pd.DataFrame(r2_scores).T.sort_values(by="r2", ascending=False)
all_scores.to_csv("output/all_scores.csv", index=False, sep=";")
print(all_scores)

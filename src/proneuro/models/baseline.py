"""Stage 1 baseline models with cross-validation.

- Amyloid: logistic regression or gradient boosting -> AUC
- Conversion: (ridge) Cox proportional hazards or gradient-boosted survival -> C-index

model="linear" (default) is unchanged from the first baseline. Use model="ridge" for
many correlated features, and model="boost" for non-linear gradient boosting.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def _classifier(model: str, seed: int):
    if model == "boost":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, random_state=seed)
    if model == "ridge":
        clf = LogisticRegressionCV(Cs=10, cv=3, max_iter=3000, scoring="roc_auc")
    else:
        clf = LogisticRegression(max_iter=1000)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def _survival(model: str, seed: int):
    if model == "boost":
        est = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                               max_depth=3, subsample=0.8, random_state=seed)
    else:
        est = CoxPHSurvivalAnalysis(alpha=50.0 if model == "ridge" else 0.1)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def amyloid_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42,
               model: str = "linear") -> dict:
    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)
    model = _classifier(model, seed)
    aucs = []
    for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        aucs.append(roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]))
    return {"n": int(len(y)), "positive_rate": float(y.mean()),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def conversion_cv(X: pd.DataFrame, surv: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                  model: str = "linear") -> dict:
    kind = model
    data = X.join(surv, how="inner")
    feats = list(X.columns)
    cidx = []
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(data):
        tr, te = data.iloc[train], data.iloc[test]
        model = _survival(kind, seed)
        model.fit(tr[feats], Surv.from_arrays(tr["event"].astype(bool), tr["duration"]))
        risk = model.predict(te[feats])
        cidx.append(concordance_index_censored(te["event"].astype(bool), te["duration"], risk)[0])
    return {"n": int(len(data)), "events": int(data["event"].sum()),
            "cindex_mean": float(np.mean(cidx)), "cindex_std": float(np.std(cidx))}

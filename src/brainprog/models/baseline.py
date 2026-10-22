"""Stage 1 baseline models with cross-validation.

- Amyloid: logistic regression -> AUC
- Conversion: Cox proportional hazards -> concordance index (C-index)
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def amyloid_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> dict:
    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    aucs = []
    for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        aucs.append(roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]))
    return {"n": int(len(y)), "positive_rate": float(y.mean()),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def conversion_cv(X: pd.DataFrame, surv: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    data = X.join(surv, how="inner")
    feats = list(X.columns)
    cidx = []
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(data):
        tr, te = data.iloc[train], data.iloc[test]
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), CoxPHSurvivalAnalysis(alpha=0.1)
        )
        model.fit(tr[feats], Surv.from_arrays(tr["event"].astype(bool), tr["duration"]))
        risk = model.predict(te[feats])
        cidx.append(concordance_index_censored(te["event"].astype(bool), te["duration"], risk)[0])
    return {"n": int(len(data)), "events": int(data["event"].sum()),
            "cindex_mean": float(np.mean(cidx)), "cindex_std": float(np.std(cidx))}

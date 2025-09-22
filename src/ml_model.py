\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from joblib import dump

FEATURES = [
    "rsi","sma_fast","sma_slow","qqq_trend_up","corr_qqq",
    "sentiment","sentiment_smooth","fng","fng_z","ret1"
]

def make_labels(df, horizon=4):
    fut_ret = df["close"].pct_change(horizon).shift(-horizon)
    y = (fut_ret > 0).astype(int)  # 1 = up, 0 = down/flat
    return y

def train_model(df, cfg, model_path):
    horiz = cfg["ml"].get("horizon_hours", 4)
    feats = df.copy()
    y = make_labels(feats, horizon=horiz).dropna()
    X = feats.loc[y.index, [c for c in FEATURES if c in feats.columns]].fillna(0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["ml"].get("test_size",0.2), random_state=cfg["ml"].get("random_state",42), shuffle=False)
    clf = GradientBoostingClassifier(random_state=cfg["ml"].get("random_state",42))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    dump(clf, model_path)
    return clf

def predict_proba_row(model, row):
    if model is None:
        return None
    import pandas as pd
    import numpy as np

    # Eğitimdeki isim sırasını koru; yoksa FEATURES kesişimini kullan
    cols = list(getattr(model, "feature_names_in_", [c for c in FEATURES if c in row.index]))

    # Satırdan güvenli çekiş + NaN/Inf temizliği
    vals = []
    for c in cols:
        v = row.get(c, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if not np.isfinite(v):  # NaN, +Inf, -Inf
            v = 0.0
        vals.append(v)

    # Çekirdek, tüm değerleri kesinlikle sonlu hale getir
    arr = np.array([vals], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    x = pd.DataFrame(arr, columns=cols)
    proba = model.predict_proba(x)[0]
    return {1: float(proba[1]), -1: float(proba[0])}

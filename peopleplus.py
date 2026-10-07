"""PeoplePlus Pulse core logic: simulation, models, decision rule, evaluation, drift (PSI)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

FEATURES = ["tenure_years","salary_ratio","months_since_promo","engagement","manager_score",
            "overtime_hrs","commute_km","training_hrs","perf_rating","remote_days"]
DEFAULTS = dict(action_cost=60000.0, effect=0.30, repl_share=0.75)

def gen(n, period, seed, target):
    r = np.random.default_rng(seed)
    ten = np.clip(r.gamma(2, 2.2, n), .1, 25); sal = np.exp(r.normal(np.log(800000), .5, n))
    sr = np.clip(r.normal(1.0 if period == 1 else .88, .15, n), .5, 1.5)
    msp = np.minimum(r.exponential(30, n), ten*12)
    eng = np.clip(r.beta(5, 3, n)*100, 5, 100); mgr = np.clip(r.normal(3.5, .8, n), 1, 5)
    ot = np.clip(r.gamma(2, 4, n), 0, 40); com = np.clip(r.gamma(2, 7, n), 1, 80); tr = np.clip(r.gamma(3, 6, n), 0, 80)
    perf = r.choice([1, 2, 3, 4, 5], n, p=[.05, .15, .45, .25, .10])
    rem = np.clip(r.poisson(1.2 if period == 1 else 2.8, n), 0, 5)
    z = (-3.5*(sr-1)*(1 if period == 1 else 2.2) - .045*(eng-60) - .5*(mgr-3.5) + .05*ot
         + (.02 if period == 1 else .004)*com + .012*msp*(ten > 1.5) + .6*(ten < 1.5) - .01*tr
         + (0 if period == 1 else 1.3*((rem >= 3) & (mgr < 3.5))))
    lo, hi = -8, 4
    for _ in range(40):
        m = (lo+hi)/2; pr = 1/(1+np.exp(-(z+m)))
        lo, hi = (m, hi) if pr.mean() < target else (lo, m)
    y = (r.random(n) < pr).astype(int)
    return pd.DataFrame(dict(tenure_years=ten, salary_ratio=sr, months_since_promo=msp, engagement=eng,
        manager_score=mgr, overtime_hrs=ot, commute_km=com, training_hrs=tr, perf_rating=perf,
        remote_days=rem, salary=sal, y=y))

def load_data():
    d1 = gen(30000, 1, 1, .14); d2 = gen(15000, 2, 2, .19)
    return dict(train=d1.iloc[:18000], val=d1.iloc[18000:24000], test=d1.iloc[24000:],
                retrain=d2.iloc[:7500], test2=d2.iloc[7500:], d2=d2)

def _gb():
    return HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=.06,
                                          l2_regularization=1.0, random_state=0)

def train(data):
    gb = _gb().fit(data["train"][FEATURES], data["train"].y)
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(data["train"][FEATURES], data["train"].y)
    gb2 = _gb().fit(data["retrain"][FEATURES], data["retrain"].y)
    return dict(gb=gb, lg=lg, gb2=gb2)

def repl_cost(d, share=.75): return share * d.salary.values

def ev_flag(d, p, action_cost, effect, share=.75):
    return p * effect * repl_cost(d, share) > action_cost

def net_benefit(d, flag, action_cost, effect, share=.75, per=1000):
    v = flag * (d.y.values * effect * repl_cost(d, share) - action_cost)
    return v.sum() / len(d) * per

def policy_stats(d, flag, action_cost, effect, share=.75):
    y = d.y.values
    return dict(net=net_benefit(d, flag, action_cost, effect, share),
                reached=(flag & (y == 1)).sum() / max(y.sum(), 1) * 100,
                actions=flag.mean() * 1000, unneeded=(flag & (y == 0)).sum() / len(d) * 1000)

def best_flat_cutoff(val, pv, action_cost=60000.0, effect=.3, share=.75):
    ths = np.linspace(.02, .9, 89)
    return max(ths, key=lambda t: net_benefit(val, pv >= t, action_cost, effect, share))

def policy_table(d, pt, pl, best_t, action_cost, effect, share):
    pol = {"No action": np.zeros(len(d), bool), "Act on everyone": np.ones(len(d), bool),
           "Hand-written rules (engagement<52, >24 months since promotion)": ((d.engagement < 52) & (d.months_since_promo > 24)).values,
           "Gradient boosting, default 0.5 cut-off": pt >= .5,
           f"Gradient boosting, tuned flat cut-off ({best_t:.2f})": pt >= best_t,
           "Logistic regression + expected-value rule": ev_flag(d, pl, action_cost, effect, share),
           "Gradient boosting + expected-value rule": ev_flag(d, pt, action_cost, effect, share)}
    rows = []
    for k, f in pol.items():
        s = policy_stats(d, f, action_cost, effect, share)
        rows.append({"Policy": k, "Net benefit per 1,000 employees (INR)": round(s["net"]),
                     "Leavers reached (%)": round(s["reached"], 1), "Actions per 1,000": round(s["actions"]),
                     "Unneeded actions per 1,000": round(s["unneeded"])})
    return pd.DataFrame(rows)

def net_vs_cutoff(d, p, action_cost, effect, share):
    ths = np.linspace(.02, .9, 89)
    return ths, [net_benefit(d, p >= t, action_cost, effect, share) for t in ths]

def decile_rates(d, p):
    dec = 9 - pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values
    return [d.y.values[dec == k].mean() * 100 for k in range(10)]

def psi(a, b, bins=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins+1)))
    if len(e) < 3:
        cats = np.unique(np.concatenate([a, b]))
        pa = np.array([(a == c).mean() for c in cats]) + 1e-4; pb = np.array([(b == c).mean() for c in cats]) + 1e-4
    else:
        e[0], e[-1] = -np.inf, np.inf
        pa = np.histogram(a, e)[0]/len(a) + 1e-4; pb = np.histogram(b, e)[0]/len(b) + 1e-4
    return float(((pb-pa)*np.log(pb/pa)).sum())

def psi_table(train, new):
    t = pd.DataFrame({"feature": FEATURES, "PSI": [psi(train[f].values, new[f].values) for f in FEATURES]})
    t["status"] = np.where(t.PSI > .25, "MAJOR shift", np.where(t.PSI > .10, "watch", "stable"))
    return t.sort_values("PSI", ascending=False).reset_index(drop=True)

def explain_employee(model, row, train, k=3):
    """Local drivers: change in risk when each feature is reset to the training median."""
    base = model.predict_proba(row[FEATURES])[0, 1]; out = []
    for f in FEATURES:
        r2 = row[FEATURES].copy(); r2[f] = train[f].median()
        out.append((f, base - model.predict_proba(r2)[0, 1]))
    return base, sorted(out, key=lambda x: -abs(x[1]))[:k]

def global_importance(model, test):
    pi = permutation_importance(model, test[FEATURES], test.y, scoring="roc_auc", n_repeats=5, random_state=0)
    return sorted(zip(FEATURES, pi.importances_mean), key=lambda x: -x[1])

def model_quality(d, p): return roc_auc_score(d.y, p), average_precision_score(d.y, p)

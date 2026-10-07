import pandas as pd, streamlit as st
import matplotlib.pyplot as plt
import peopleplus as pp

st.set_page_config(page_title="PeoplePlus Pulse", page_icon="👥", layout="wide")
G, N, RD = "#1f9d8b", "#1f3a5f", "#d1495b"

@st.cache_resource
def setup():
    data = pp.load_data(); models = pp.train(data)
    pv = models["gb"].predict_proba(data["val"][pp.FEATURES])[:, 1]
    return data, models, pp.best_flat_cutoff(data["val"], pv)
data, models, best_t = setup()
te = data["test"]; pt = models["gb"].predict_proba(te[pp.FEATURES])[:, 1]; pl = models["lg"].predict_proba(te[pp.FEATURES])[:, 1]

st.title("PeoplePlus Pulse")
st.caption("Cost-aware attrition prediction and retention decisioning with drift monitoring | Team26 | Hackathon 4.0 Track 2. All data is simulated.")
tab1, tab2, tab3 = st.tabs(["Retention simulator", "Employee scorer", "Drift monitor"])

with st.sidebar:
    st.header("Economics (inputs)")
    cost = st.slider("Retention action cost (INR)", 10000, 200000, 60000, 5000)
    eff = st.slider("Reduction in chance of leaving", .1, .8, .30, .05)
    share = st.slider("Replacement cost (share of annual pay)", .3, 1.5, .75, .05)

with tab1:
    st.subheader("Which policy earns the most?")
    tbl = pp.policy_table(te, pt, pl, best_t, cost, eff, share)
    st.dataframe(tbl, hide_index=True)
    best = tbl.iloc[:, 1].idxmax(); st.success(f"Best policy under these inputs: **{tbl.iloc[best, 0]}**")
    st.markdown("**Expected-value rule:** act when `p x effect x replacement cost > action cost`. Risk bar by annual pay: " +
                ", ".join(f"INR {s} lakh: {cost/(eff*share*s*1e5):.0%}" for s in (3, 8, 20)) + ".")
    ths, nets = pp.net_vs_cutoff(te, pt, cost, eff, share)
    ev = pp.policy_stats(te, pp.ev_flag(te, pt, cost, eff, share), cost, eff, share)["net"]
    fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(ths, nets, color=G, label="Flat cut-off"); ax.axhline(ev, ls="--", color=N, label="Expected-value rule")
    ax.axvline(.5, ls=":", color=RD); ax.set_xlabel("Act if attrition risk >= cut-off"); ax.set_ylabel("Net benefit per 1,000 (INR)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False); st.pyplot(fig)
    auc, pr = pp.model_quality(te, pt); st.caption(f"Gradient boosting on held-out data: ROC-AUC {auc:.2f}, PR-AUC {pr:.2f}.")
    imp = pp.global_importance(models["gb"], te)[:4]; st.caption("Strongest signals (permutation importance): " + ", ".join(f"{f}" for f, _ in imp) + ".")

with tab2:
    st.subheader("Score a single employee")
    med = data["train"][pp.FEATURES].median(); cols = st.columns(5); vals = {}
    rng = {"tenure_years": (.1, 25., float(med.tenure_years)), "salary_ratio": (.5, 1.5, float(med.salary_ratio)),
           "months_since_promo": (0., 120., float(med.months_since_promo)), "engagement": (5., 100., float(med.engagement)),
           "manager_score": (1., 5., float(med.manager_score)), "overtime_hrs": (0., 40., float(med.overtime_hrs)),
           "commute_km": (1., 80., float(med.commute_km)), "training_hrs": (0., 80., float(med.training_hrs)),
           "perf_rating": (1., 5., 3.), "remote_days": (0., 5., float(round(med.remote_days)))}
    for i, (f, (lo, hi, dv)) in enumerate(rng.items()): vals[f] = cols[i % 5].slider(f, lo, hi, dv)
    sal = st.slider("Annual pay (INR lakh)", 2.0, 40.0, 8.0, .5)
    row = pd.DataFrame([vals]); p, drivers = pp.explain_employee(models["gb"], row, data["train"])
    bar = cost / (eff * share * sal * 1e5); act = p > bar
    a, b = st.columns(2); a.metric("Attrition risk (12 months)", f"{p:.1%}"); b.metric("Risk bar at this pay", f"{bar:.1%}")
    (st.error if act else st.success)(f"Decision: **{'ACT: retention conversation / pay review' if act else 'No action needed'}**")
    st.write("Signals behind the score (change in risk vs a typical employee):")
    for f, d_ in drivers: st.write(f"- `{f}` = {vals[f]:.1f}: {d_:+.1%}")
    st.caption("Scores support HR conversations; they should never be used for punitive decisions.")

with tab3:
    st.subheader("Has the data drifted?")
    t = pp.psi_table(data["train"], data["d2"]); st.dataframe(t.style.format({"PSI": "{:.2f}"}), hide_index=True)
    major = t[t.PSI > .25].feature.tolist()
    if major: st.error("Major drift in: " + ", ".join(major) + ". Retraining recommended.")
    else: st.success("No major drift.")
    t2 = data["test2"]; s_p = models["gb"].predict_proba(t2[pp.FEATURES])[:, 1]; r_p = models["gb2"].predict_proba(t2[pp.FEATURES])[:, 1]
    ss = pp.policy_stats(t2, pp.ev_flag(t2, s_p, cost, eff, share), cost, eff, share); rs = pp.policy_stats(t2, pp.ev_flag(t2, r_p, cost, eff, share), cost, eff, share)
    x, y = st.columns(2); x.metric("Stale model, net per 1,000 (INR)", f"{ss['net']:,.0f}", f"{ss['actions']:.0f} actions per 1,000", delta_color="off")
    y.metric("Retrained model, net per 1,000 (INR)", f"{rs['net']:,.0f}", f"{rs['net']/ss['net']-1:+.0%} vs stale")

"""Regenerates report-style figures into ./figures"""
import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import peopleplus as pp
os.makedirs("figures", exist_ok=True)
d = pp.load_data(); m = pp.train(d); te = d["test"]; pt = m["gb"].predict_proba(te[pp.FEATURES])[:, 1]
c, e, s = pp.DEFAULTS["action_cost"], pp.DEFAULTS["effect"], pp.DEFAULTS["repl_share"]
ths, nets = pp.net_vs_cutoff(te, pt, c, e, s); ev = pp.policy_stats(te, pp.ev_flag(te, pt, c, e, s), c, e, s)["net"]
plt.figure(figsize=(6, 3)); plt.plot(ths, nets); plt.axhline(ev, ls="--"); plt.xlabel("Cut-off"); plt.ylabel("Net benefit per 1,000 (INR)"); plt.tight_layout(); plt.savefig("figures/net_vs_cutoff.png", dpi=150)
plt.figure(figsize=(6, 3)); plt.bar(range(1, 11), pp.decile_rates(te, pt)); plt.xlabel("Risk decile"); plt.ylabel("Attrition (%)"); plt.tight_layout(); plt.savefig("figures/deciles.png", dpi=150)
t = pp.psi_table(d["train"], d["d2"]); plt.figure(figsize=(6, 3)); plt.barh(t.feature[::-1], t.PSI[::-1]); plt.axvline(.25, ls="--"); plt.xlabel("PSI"); plt.tight_layout(); plt.savefig("figures/psi.png", dpi=150)
print("Saved figures/")

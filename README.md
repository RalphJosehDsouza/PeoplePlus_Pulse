# PeoplePlus Pulse

**Cost-aware attrition prediction and retention decisioning with explainable drivers and drift monitoring**
Team26 | Hackathon 4.0 | Track 2: PeoplePlus AI | IES MCRC Avenir Analytics Club

Team: Sean Pereira, Pranay Reddy, Ralph Dsouza (B.E. Computer Engineering, Fr. Conceicao Rodrigues College of Engineering)

| | |
|---|---|
| Live demo | `https://peoplepluspulse-ztaxggzhuaoiddzvvrdewg.streamlit.app/` |
| Report | `Team26_PeoplePlusAI_Hackathon4.0.pdf` |

<!-- Add a screenshot or GIF here: ![demo](figures/demo.gif) -->

## The idea
Predicting who might leave is not the same as deciding whom to act on. PeoplePlus Pulse acts on an employee only when the expected cost of losing them exceeds the cost of a retention action:

`p(leave) x effect x replacement cost > action cost`

so senior, high-cost roles are flagged at a lower risk score than junior ones. The dashboard also monitors drift (PSI) and shows the value of retraining.

## Results (simulated data, 6,000 held-out employees)
| Policy | Net benefit per 1,000 employees (INR) |
|---|---|
| Act on everyone | -31,784,319 |
| Hand-written rules | 101,274 |
| Default 0.5 cut-off | 1,244,463 |
| Tuned flat cut-off | 2,908,226 |
| **Gradient boosting + expected-value rule** | **4,246,189** |

After drift, retraining lifts net benefit from 8,862,813 to 11,352,694 (+28%). Logistic regression is on par with gradient boosting under the same rule (4,341,137); most of the gain comes from the decision rule.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Optional: `python make_figures.py` saves charts to `figures/`.

## Files
- `peopleplus.py`: simulation, models, decision rule, evaluation, PSI
- `app.py`: Streamlit dashboard (Retention simulator, Employee scorer, Drift monitor)
- `make_figures.py`: regenerates charts

## Limitations
All data is simulated; real attrition is noisier. The INR 60,000 action cost, 30% effectiveness and 75% replacement-cost share are assumptions to validate in a shadow-mode pilot. Scores support HR conversations and must not be used for punitive decisions; handle real employee data under India's DPDP Act, 2023.

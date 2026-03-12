# Financial Time-Series Predictor

Hybrid LightGBM + Ridge model for financial time-series forecasting and tactical asset allocation, with a Streamlit UI.

![Banner](assets/banner.png)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/github/license/Simarbir2112006/financial-time-series-predictor)
![Last Commit](https://img.shields.io/github/last-commit/Simarbir2112006/financial-time-series-predictor)
![Stars](https://img.shields.io/github/stars/Simarbir2112006/financial-time-series-predictor?style=social)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)](https://financial-time-series-predictor.streamlit.app/)

---

## Overview

LightGBM captures nonlinear regime-dependent patterns; Ridge provides a stable low-bias baseline. Signals feed into a volatility-aware allocation layer that normalises position sizes to consistent portfolio risk. A rolling-window pipeline handles walk-forward training and out-of-sample inference throughout.

---

## Walkthrough

https://github.com/user-attachments/assets/walkthrough.mp4

---

## Project Structure

```
.
├── artifacts/
│   ├── hull_model.pkl          # serialised model for demo runs
│   └── selected_features.pkl
├── assets/
│   └── banner.png
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── demo_market_cycle.csv
│   └── bull_market_test.csv
├── src/
│   ├── config.py               # centralised configuration
│   ├── model.py                # hybrid model definition
│   ├── pipeline.py             # ingestion, features, walk-forward logic
│   ├── train.py                # training entrypoint
│   ├── predict.py              # batch inference
│   └── utils.py                # metrics and IO helpers
├── synthetic_data_generator/
│   ├── generate_bull_market.py
│   └── generate_demo_cycle.py
├── ui/
│   └── app.py                  # Streamlit dashboard
└── requirements.txt
```

---

## Architecture

```
Raw Market Data (data/*.csv)
        │
        ▼
[ pipeline.py ] — rolling-window ingestion + feature engineering
        │
        ▼
[ model.py ] — hybrid training (Ridge trend + LightGBM residuals)
        │
        ▼
[ utils.py ] — volatility-aware allocation
        │
        ▼
[ ui/app.py ] — Streamlit visualisation + CSV export
```

---

## Quick Start

```bash
git clone https://github.com/Simarbir2112006/financial-time-series-predictor.git
cd financial-time-series-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train**
```bash
python -m src.train
```

**Predict**
```bash
python -m src.predict
```

**Streamlit UI**
```bash
streamlit run ui/app.py
```

**Generate synthetic data**
```bash
cd synthetic_data_generator
python generate_demo_cycle.py
python generate_bull_market.py
```

---

## UML

![Pipeline UML](assets/financial-time-series-predictor-pipleine.drawio.png)

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `Streamlit`

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Simarbir Singh Sandhu**  
[GitHub](https://github.com/Simarbir2112006) · [LinkedIn](https://www.linkedin.com/in/simarbir-singh-sandhu/) · [X](https://x.com/sandhusimarbir)

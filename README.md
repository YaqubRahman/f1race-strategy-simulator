# 🏎️ F1 Strategy Predictor

A machine learning powered tool that predicts the optimal pit stop strategy for a Formula 1 race. Select a driver and circuit, and the model recommends when to pit and which tyre compounds to run based on historical lap data.

<img width="322" height="468" alt="Screenshot 2026-06-09 222711" src="https://github.com/user-attachments/assets/bb9490fc-b533-43c6-8c12-d3f4ae665dd3" />

## How it works

1. Historical race data is loaded from the [FastF1](https://docs.fastf1.dev/) library for the selected driver and circuit (2018–2024)
2. A Random Forest regression model is trained on lap time features including tyre compound, tyre life, lap number, and lap delta
3. A strategy simulation loop tests every possible pit lap and compound combination
4. The combination with the lowest predicted total race time is returned as the optimal strategy

## Tech stack

- **Frontend** — React, TypeScript, Tailwind CSS, Vite
- **Backend** — Python, Flask
- **ML** — scikit-learn (Random Forest Regressor)
- **Data** — FastF1, pandas

## Getting started

### Backend

```bash
pip install flask flask-cors fastf1 scikit-learn pandas
python app.py
```

Flask will start on `http://localhost:5000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Limitations

- Assumes dry weather racing, wet conditions and intermediate tyres are not modelled
- Predictions are based on one driver's historical data at a given circuit, so results are more reliable for drivers with more race history there
- Tyre degradation is approximated using average lap deltas per compound rather than a full physical degradation model
- Training and prediction happens on each request which adds latency, model caching would improve response times

## Future work

- Add weather and safety car probability modelling
- Support multi-stop strategies
- Cache trained models to reduce prediction time
- Expand driver and circuit coverage

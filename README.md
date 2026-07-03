# ⚽ FIFA World Cup 2026 Match Predictor
### Australia 🇦🇺 vs Egypt 🇪🇬 (Round of 32)

A machine learning project that predicts the outcome of the FIFA World Cup 2026 Round of 32 match between **Australia** and **Egypt** using an ensemble of statistical and machine learning models.

The project combines historical match simulation, feature engineering, classical ML algorithms, Poisson goal modeling, and probability calibration to estimate:

- 90-minute match result probabilities
- Knockout winner probabilities
- Most likely scorelines
- Expected Goals (xG)
- Mohamed Salah fitness sensitivity analysis
- Interactive prediction dashboard

---

## Project Overview

This repository demonstrates how multiple predictive techniques can be combined to forecast football matches.

The prediction pipeline consists of:

```
Historical Dataset
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Models
(Logistic Regression
 Random Forest
 XGBoost
 Gradient Boosting
 Neural Network Simulation)
        │
        ▼
Weighted Ensemble
        │
        ▼
Poisson Goal Model
        │
        ▼
Final Prediction
        │
        ▼
Visualization Dashboard
```

---

## Models Used

The project combines predictions from:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Simulated Neural Network
- Dixon-Coles Poisson Goal Model

Each ML model is evaluated using cross-validation, and ensemble weights are assigned according to model performance.

---

## Feature Engineering

The model extracts numerous football-specific features including:

### Team Form

- Recent points
- Win rate
- Draw rate
- Goals scored
- Goals conceded
- Clean sheet rate
- Goal difference
- Expected Goals (xG)

### FIFA Information

- FIFA rankings
- Ranking difference

### Head-to-Head Statistics

- Historical win rate
- Draw rate
- Average goals

### World Cup Context

- Group stage points
- Goal difference
- Goals scored
- Goals conceded
- Knockout stage indicator

---

## Statistical Model

Besides machine learning, the project uses a **Poisson Goal Model** to estimate:

- Expected goals
- Scoreline probabilities
- Win / Draw / Loss probabilities

This allows realistic football score predictions instead of only class probabilities.

---

## Ensemble Strategy

Final predictions combine:

- 60% Machine Learning Ensemble
- 40% Poisson Model

The project also converts 90-minute probabilities into knockout-stage advancement probabilities by redistributing draw outcomes into extra time and penalties.

---

## Sensitivity Analysis

A dedicated section studies how **Mohamed Salah's fitness** influences match predictions.

Scenarios include:

- Fully fit
- 80% fit
- Limited minutes
- Absent

The analysis shows how attacking strength affects Egypt's probability of advancing.

---

## Visual Dashboard

The generated dashboard includes:

- Model comparison
- Final probability donut chart
- Scoreline heatmap
- XGBoost feature importance
- Radar comparison
- Group stage statistics
- Advancement probabilities
- Salah fitness sensitivity analysis

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- Matplotlib
- Seaborn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/world-cup-predictor.git
cd world-cup-predictor
```

Install dependencies:

```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn
```

Run the predictor:

```bash
python predictor.py
```

---

## Output

The program produces:

- Model probabilities
- Cross-validation metrics
- Expected goals
- Most likely scorelines
- Match winner probabilities
- Sensitivity analysis
- Prediction dashboard image

---

## Example Outputs

- Australia Win Probability
- Draw Probability
- Egypt Win Probability
- Australia Advancement Probability
- Egypt Advancement Probability
- Top 10 Most Likely Scorelines

---

## Notes

This project is intended for educational and research purposes.

Although the model incorporates real FIFA World Cup 2026 group-stage results, the historical dataset used for training is partially simulated to demonstrate a complete end-to-end machine learning workflow.

Football remains inherently unpredictable, and no model can guarantee actual match outcomes.

---

## Future Improvements

- Real historical international match database
- FIFA Elo ratings
- Player-level statistics
- Team market values
- Injury database integration
- Live betting odds
- Bayesian updating
- Deep Neural Networks
- Transformer-based sequence models
- SHAP explainability
- Automated web data collection

---

## Author

**Ghaith Hajji**

Business Analytics Student  
Machine Learning & Artificial Intelligence Enthusiast

---

## License

This project is released under the MIT License.

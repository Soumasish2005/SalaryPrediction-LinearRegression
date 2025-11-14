# SalaryPrediction-LinearRegression

Project to predict employee salaries using supervised regression models and ensemble techniques (Linear Regression, Random Forest, Gradient Boosting, Voting Regressor). The project demonstrates end-to-end steps: data loading, cleaning, encoding, training, evaluation, visualization, and saving the best model.

## Key Features
- Data cleaning and simple feature engineering
- Label encoding for categorical fields
- Multiple regressors: Linear Regression, Random Forest, Gradient Boosting
- Model ensembling with Voting Regressor
- Model evaluation (MSE, RMSE, MAE, R²)
- Visualizations: correlation heatmap, actual vs predicted, feature importances
- Save best-performing model for deployment

## Dataset
The notebook expects a CSV named `Salary Data.csv` placed in the project root. The dataset should include salary and predictive features such as job title, experience, education, location, etc.

(If dataset originated from Kaggle, ensure you follow the dataset's license and attribution requirements.)

## Quickstart / Setup
Recommended: use the provided dev container (Ubuntu 24.04). From the project root:

1. Create or activate a Python environment:
   - ```python -m venv nbenv && source nbenv/bin/activate```

2. Start the Jupyter notebook:
   - ```jupyter notebook```
   - or open the notebook file: `Salary_Prediction_v1.ipynb`

## Usage
- Open `Salary_Prediction_v1.ipynb` and run cells sequentially.
- The notebook:
  - Loads `Salary Data.csv`
  - Drops missing values
  - Encodes categorical columns with LabelEncoder (saved encoders in-memory)
  - Splits data into train/test
  - Trains models and selects the best by R²
  - Saves the best model to `./saved_models/best_salary_model.joblib`
  - Displays evaluation metrics and plots

Example: after training the best model is saved as:
- `./saved_models/best_salary_model.joblib`

You can load it for inference:
```python
import joblib
model = joblib.load("./saved_models/best_salary_model.joblib")
pred = model.predict(X_new)
```

## Evaluation & Visuals
The notebook outputs:
- A summary table of model metrics (MSE, RMSE, MAE, R²)
- Correlation heatmap for feature analysis
- Actual vs Predicted plot for the best model
- Barplot comparison of R² across models
- Feature importances for Random Forest (if trained)

## Project Structure
- Salary_Prediction_v1.ipynb    — Main analysis and training notebook
- Salary Data.csv               — Input dataset (not included in repo)
- saved_models/                 — Directory to store the serialized best model
- README.md                     — This file

## Recommendations / Next Steps
- Add a requirements.txt or environment.yml for reproducibility.
- Persist LabelEncoders and preprocessing pipeline (e.g., with sklearn.pipeline) to ensure consistent inference transformations.
- Perform cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Expand feature engineering (text/vector features for job titles, location encoding, experience bucketing).
- Consider model explainability tools (SHAP/LIME) for feature impact analysis.

## Contributing
Contributions are welcome. Please open an issue or submit a pull request with a clear description of changes.

## License
Add a license file (e.g., MIT) that suits your project's needs. If using Kaggle or third-party datasets, ensure compliance with their license and attribution requirements.
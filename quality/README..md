# 🍷 Wine Quality Prediction (Regression)

Predict wine quality using the most impactful chemical properties.

📂 Project Structure
wine-quality-prediction/
├── app.py # Streamlit web app
├── model.pkl # Trained model
├── metrics.json # Saved performance metrics
├── requirements.txt # Dependencies
├── data/
│ └── WineQT.csv # Dataset
├── notebooks/
│ └── model_training.ipynb # EDA, preprocessing, training


## 🚀 How to Run Locally
```bash
# Create virtual environment
python -m venv wine_env
# Activate
wine_env\Scripts\activate    # Windows
source wine_env/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

📊 Model
Trained using RandomForestRegressor on top 6 features.

Evaluated using RMSE & R².


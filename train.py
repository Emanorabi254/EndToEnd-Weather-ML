import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processor import WeatherDataProcessor
import os

def run_training():
    # 1. Setup Directories
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models/' directory.")

    # 2. Load Raw Data
    print("Loading raw data...")
    try:
        df_raw = pd.read_csv("weatherAUS.csv")
    except FileNotFoundError:
        print("Error: 'weatherAUS.csv' not found in the current directory.")
        return

    # 3. Process Data
    print("Starting data cleaning and feature engineering...")
    processor = WeatherDataProcessor()
    df_cleaned = processor.clean_data(df_raw, is_training=True)

    # 4. Save Processing Assets (Scaler and LabelEncoder)
    print("Saving processing assets...")
    processor.save_assets('models/')

    # 5. Split Data
    # Drop columns not used for training
    X = df_cleaned.drop(['RainTomorrow', 'Date', 'month'], axis=1)
    y = df_cleaned['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Train Model
    print("Training Random Forest Classifier (this may take a minute)...")
    # Using balanced class weight because of the imbalanced nature of rain data
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1 # Uses all CPU cores for faster training
    )
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save the Trained Model
    print("Saving the final model...")
    joblib.dump(model, 'models/rain_forest_model.pkl')
    print("✅ Training complete! All files saved in the 'models/' folder.")

if __name__ == "__main__":
    run_training()
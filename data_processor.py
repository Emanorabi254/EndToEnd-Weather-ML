import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer

class WeatherDataProcessor:
    def __init__(self):
        # --- THIS PART WAS LIKELY MISSING OR DIFFERENT ---
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()
        self.num_imputer = SimpleImputer(strategy="median")
        self.knn_imputer = KNNImputer(n_neighbors=5)
        # --------------------------------------------------
        
        self.compass_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

    def clean_data(self, df, is_training=True):
        df = df.copy()
        
        # 1. Date & Sorting
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df = df.sort_values(['Location', 'Date'])

        # 2. Median Imputation
        simple_cols = ["MinTemp", "MaxTemp", "Rainfall", "Temp9am", "Temp3pm", 
                       "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm"]
        for col in simple_cols:
            df[col] = df.groupby("Location")[col].transform(lambda x: x.fillna(x.median()))

        # 3. Time Series Imputation
        time_cols = ["WindGustSpeed", "Pressure9am", "Pressure3pm"]
        for col in time_cols:
            df[col] = df.groupby("Location")[col].transform(lambda x: x.ffill().bfill())    

        # 4. Categorical Imputation
        cat_cols = ["WindDir9am", "WindGustDir", "WindDir3pm", "RainToday"]
        for col in cat_cols:
            df[col] = df.groupby("Location")[col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
            )

        # 5. Global Imputation (Using the attributes defined in __init__)
        if is_training:
            df[time_cols] = self.num_imputer.fit_transform(df[time_cols])
        else:
            df[time_cols] = self.num_imputer.transform(df[time_cols])

        knn_cols = ["Cloud9am", "Cloud3pm", "Evaporation", "Sunshine"]
        if is_training:
            df[knn_cols] = self.knn_imputer.fit_transform(df[knn_cols])
        else:
            df[knn_cols] = self.knn_imputer.transform(df[knn_cols])

        if "RainTomorrow" in df.columns:
            df = df.dropna(subset=["RainTomorrow"])

        # 6. Outlier Capping
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.where(df[col] > Q3 + 1.5*IQR, Q3 + 1.5*IQR, 
                               np.where(df[col] < Q1 - 1.5*IQR, Q1 - 1.5*IQR, df[col]))

        # 7. Encoding (With Fix for the FutureWarning)
        map_logic = {"Yes": 1, "No": 0}
        df["RainToday"] = df["RainToday"].replace(map_logic)
        if "RainTomorrow" in df.columns:
            df["RainTomorrow"] = df["RainTomorrow"].replace(map_logic)
        
        # New Pandas versions want this to avoid the warning:
        df = df.infer_objects(copy=False)

        # 8. Location Encoding
        if is_training:
            df["Location"] = self.le.fit_transform(df["Location"])
        else:
            df["Location"] = self.le.transform(df["Location"])

        # 9. Wind/Month Cyclic Encoding
        wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        for col in wind_cols:
            df[col + '_deg'] = df[col].map(self.compass_map)
            df[col + '_sin'] = np.sin(2 * np.pi * df[col + '_deg'] / 360)
            df[col + '_cos'] = np.cos(2 * np.pi * df[col + '_deg'] / 360)
            df.drop([col, col + '_deg'], axis=1, inplace=True)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 10. Feature Engineering
        df["Pressure_Diff"] = df["Pressure3pm"] - df["Pressure9am"]
        df["Humidity_Diff"] = df["Humidity3pm"] - df["Humidity9am"]
        df["WindSpeed_Diff"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
        df['Cloud_Total'] = df['Cloud9am'] + df['Cloud3pm']

        # 11. Scaling
        if is_training:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def save_assets(self, path="models/"):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.le, f"{path}location_encoder.pkl")
        joblib.dump(self.scaler, f"{path}scaler.pkl")
        joblib.dump(self.num_imputer, f"{path}num_imputer.pkl")
        joblib.dump(self.knn_imputer, f"{path}knn_imputer.pkl")
        print(f"✅ Assets saved to {path}")

    def load_assets(self, path="models/"):
        self.le = joblib.load(f"{path}location_encoder.pkl")
        self.scaler = joblib.load(f"{path}scaler.pkl")
        self.num_imputer = joblib.load(f"{path}num_imputer.pkl")
        self.knn_imputer = joblib.load(f"{path}knn_imputer.pkl")
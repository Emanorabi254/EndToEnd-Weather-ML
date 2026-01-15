import pandas as pd
from data_processor import WeatherDataProcessor

processor = WeatherDataProcessor()
raw_df = pd.read_csv("weatherAUS.csv")
print("Cleaning data... please wait (this takes a moment due to KNN)")
df = processor.clean_data(raw_df, is_training=True)
df.to_csv("weather_cleaned_for_dash.csv", index=False)
print("✅ Cleaned data saved!")
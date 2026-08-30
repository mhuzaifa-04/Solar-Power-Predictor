import os
import pandas as pd
import numpy as np

def load_and_clean_solar_data(gen_file_path, weather_file_path):
    print("⏳ Loading raw CSV files...")
    gen_df = pd.read_csv(gen_file_path)
    weather_df = pd.read_csv(weather_file_path)

    # 1. Standardize Timestamp Formats
    print("🔄 Standardizing timestamp formats...")
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # 2. Rename IRRADIATION to IRRADIANCE if present
    if 'IRRADIATION' in weather_df.columns:
        weather_df = weather_df.rename(columns={'IRRADIATION': 'IRRADIANCE'})

    # 3. Group Generation Data by Timestamp
    print("📊 Aggregating generation output by timestamp...")
    gen_grouped = gen_df.groupby('DATE_TIME').agg({
        'DC_POWER': 'mean',
        'AC_POWER': 'mean',
        'DAILY_YIELD': 'max'
    }).reset_index()

    # 4. Merge Generation and Weather Data
    print("🔗 Merging generation and weather data...")
    merged_df = pd.merge(gen_grouped, weather_df, on='DATE_TIME', how='inner')

    # 5. Feature Engineering
    print("🛠️ Extracting temporal and physical features...")
    merged_df['HOUR'] = merged_df['DATE_TIME'].dt.hour
    merged_df['MONTH'] = merged_df['DATE_TIME'].dt.month
    merged_df['DAY_OF_WEEK'] = merged_df['DATE_TIME'].dt.dayofweek
    
    # Calculate Panel Heating Overhead
    merged_df['TEMP_DIFF'] = merged_df['MODULE_TEMPERATURE'] - merged_df['AMBIENT_TEMPERATURE']

    # 6. Clean Missing & Unrealistic Values
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df['AC_POWER'] >= 0]

    print(f"✅ Success! Cleaned dataset shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    gen_path = 'data/Plant_1_Generation_Data.csv'
    weather_path = 'data/Plant_1_Weather_Sensor_Data.csv'

    if os.path.exists(gen_path) and os.path.exists(weather_path):
        processed_data = load_and_clean_solar_data(gen_path, weather_path)
        processed_data.to_csv('data/cleaned_solar_data.csv', index=False)
        print("💾 Saved processed file to: data/cleaned_solar_data.csv")
    else:
        print("❌ CSV files missing! Place Plant_1_Generation_Data.csv and Plant_1_Weather_Sensor_Data.csv inside the data/ folder.")
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from itertools import permutations


def convert(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}mins:{remaining_seconds:.3f}s"


print(fastf1.__version__)
fastf1.Cache.enable_cache('cache')

driver = 'VER' 
track = 'Silverstone'
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
all_laps = []

for year in years:
    try:
        session = fastf1.get_session(year, track, 'R')  
        session.load()
        laps = session.laps.pick_drivers([driver])

        # Lap times in seconds
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

        laps_clean = laps[(laps['LapTimeSeconds'].notna()) & 
                          (laps['SpeedFL'].notna()) & 
                          (laps['PitInTime'].isna()) & 
                          (laps['Deleted'] != True) & 
                          (laps['IsAccurate'] == True)]

        # Sector times in seconds
        laps['Sector1Seconds'] = laps['Sector1Time'].dt.total_seconds()
        laps['Sector2Seconds'] = laps['Sector2Time'].dt.total_seconds()
        laps['Sector3Seconds'] = laps['Sector3Time'].dt.total_seconds()

        laps_clean['Year'] = year

        all_laps.append(laps_clean)

    except Exception as e:
        print(f"Could not load data for {year} {track}: {e}")
        continue

# Combine all years into one DataFrame
multi_year_laps = pd.concat(all_laps, ignore_index=True)

print(multi_year_laps.head())
print("Total laps collected:", len(multi_year_laps))



# Selecting relevant features and the target variable
features = multi_year_laps[["LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre", "SpeedFL", "TrackStatus"]]
print(features.head())

labels = multi_year_laps["LapTimeSeconds"]

# One-hot encoding - converting categorical features into numerical format
features = pd.get_dummies(features, columns=["Compound", "TrackStatus"])

# Imputing missing values if any
# imputer = SimpleImputer(strategy='mean')
# features_imputed = imputer.fit_transform(features)

# Splitting the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# # Standardizing the features
# ct = ColumnTransformer([
#     ('scale', StandardScaler(), ["LapNumber", "TyreLife", "SpeedFL"])
# ], remainder='passthrough')

# features_train = ct.fit_transform(features_train)
# features_test = ct.transform(features_test)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(features_train, labels_train)
print("R^2 score:", reg.score(features_test, labels_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error
preds = reg.predict(features_test)
print("MAE:", mean_absolute_error(labels_test, preds))
print("MSE:", mean_squared_error(labels_test, preds))
print("AHHHHHHHHHHHHHHHH", preds)
print(multi_year_laps["TrackStatus"].value_counts())

# plt.scatter(labels_test, labels_test - preds, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel("Actual Lap Time (s)")
# plt.ylabel("Residual (Actual - Predicted)")
# plt.title("Residuals vs Actual Lap Time")
# plt.show()



total_laps = int(session.laps['LapNumber'].max())

data_compound1 = {
    "LapNumber": list(range(0, total_laps)),
    "Stint": 1,
    "Compound": ["SOFT"] * total_laps,
    "TyreLife": list(range(1, total_laps + 1)),
    "FreshTyre": [True] + [False] * (total_laps - 1),
    "SpeedFL": [multi_year_laps["SpeedFL"].mean()] * total_laps,
    "TrackStatus": [1] * total_laps
}

data_compound2 = {
    "LapNumber": list(range(0, total_laps)),
    "Stint": 1,
    "Compound": ["MEDIUM"] * total_laps,
    "TyreLife": list(range(1, total_laps + 1)),
    "FreshTyre": [True] + [False] * (total_laps - 1),
    "SpeedFL": [multi_year_laps["SpeedFL"].mean()] * total_laps,
    "TrackStatus": [1] * total_laps
}

data_compound3 = {
    "LapNumber": list(range(0, total_laps)),
    "Stint": 1,
    "Compound": ["HARD"] * total_laps,
    "TyreLife": list(range(1, total_laps + 1)),
    "FreshTyre": [True] + [False] * (total_laps - 1),
    "SpeedFL": [multi_year_laps["SpeedFL"].mean()] * total_laps,
    "TrackStatus": [1] * total_laps
}

# Create DataFrames for each compound
compound1_df = pd.DataFrame(data_compound1)
compound2_df = pd.DataFrame(data_compound2)
compound3_df = pd.DataFrame(data_compound3)

# One-hot encoding for the new DataFrames
compound1_df = pd.get_dummies(compound1_df, columns=["Compound", "TrackStatus"])
compound2_df = pd.get_dummies(compound2_df, columns=["Compound", "TrackStatus"])
compound3_df = pd.get_dummies(compound3_df, columns=["Compound", "TrackStatus"])

# Reindex the new DataFrames to ensure they have the same columns as the training features, filling missing columns with zeros
# (MEDIUM and HARD will have 0s in the SOFT columns, etc.)
compound1_df = compound1_df.reindex(columns=features.columns, fill_value=0)
compound2_df = compound2_df.reindex(columns=features.columns, fill_value=0)
compound3_df = compound3_df.reindex(columns=features.columns, fill_value=0)

compounds = [
    ("SOFT", compound1_df),
    ("MEDIUM", compound2_df),
    ("HARD", compound3_df)
]



stint1_time = reg.predict(compound1_df)
stint1_total_time = stint1_time.sum()
print("Stint 1 Total Time (SOFT):", stint1_total_time)
stint2_time = reg.predict(compound2_df)
stint2_total_time = stint2_time.sum()
print("Stint 2 Total Time (MEDIUM):", stint2_total_time)
stint3_time = reg.predict(compound3_df)
stint3_total_time = stint3_time.sum()
print("Stint 3 Total Time (HARD):", stint3_total_time)
pit_stop_constant = 23.0


best_strategy = 0
best_time = float('inf')
best_compound1 = None
best_compound2 = None

# F1 Teams never pit before lap 10 at the earliest
for (name1, compound1_df), (name2, compound2_df) in permutations(compounds, 2):
    for pit_lap in range(10, total_laps - 10):
        stint1_time = reg.predict(compound1_df.iloc[:pit_lap])
        stint2_time = reg.predict(compound2_df.iloc[pit_lap:])

        stint1_total_time = stint1_time.sum()
        stint2_total_time = stint2_time.sum()
        total_time = stint1_total_time + stint2_total_time + pit_stop_constant

        if total_time < best_time:
            best_time = total_time
            best_strategy = pit_lap
            best_compound1 = name1
            best_compound2 = name2

print(f"Best strategy: Pit on lap {best_strategy}, {best_compound1} → {best_compound2}")
print(f"Predicted total race time: {convert(best_time)}")




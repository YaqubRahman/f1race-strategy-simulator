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


print(fastf1.__version__)
fastf1.Cache.enable_cache('cache')  
session = fastf1.get_session(2024, 'Abu Dhabi', 'R') 
session.load()


laps = session.laps.pick_driver('VER') 
print(laps.head())
print("+-----------------------+")
print(laps.columns)


laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

# Lap times in seconds
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Sector times in seconds
laps['Sector1Seconds'] = laps['Sector1Time'].dt.total_seconds()
laps['Sector2Seconds'] = laps['Sector2Time'].dt.total_seconds()
laps['Sector3Seconds'] = laps['Sector3Time'].dt.total_seconds()


# Drop rows with any NaN in features or labels
laps_clean = laps.dropna(subset=[
    "LapTimeSeconds", "SpeedFL"
])

laps_clean = laps[
    (laps['LapTimeSeconds'].notna()) &
    (laps['SpeedFL'].notna()) &
    (laps['PitInTime'].isna()) &
    (laps['Deleted'] != True) &
    (laps['IsAccurate'] == True)
]


# Selecting relevant features and the target variable
features = laps_clean[["LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre", "SpeedFL", "TrackStatus"]]
print(features.head())

labels = laps_clean["LapTimeSeconds"]

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

plt.scatter(labels_test, labels_test - preds, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Actual Lap Time")
plt.show()





# Marking the pit stop laps/spikes in the graph
#pit_in_laps = laps[laps['PitInTime'].notna()]
#pit_out_laps = laps[laps['PitOutTime'].notna()]

# Unique tire compounds used
#compounds = laps['Compound'].unique()

# Graphing lap times with pit stops and tire compounds
# plt.figure(figsize=(10,6))
# plt.plot(laps['LapNumber'], laps['LapTimeSeconds'], marker='o')
# plt.xlabel("Lap Number")
# plt.ylabel("Lap Time (s)")
# plt.title("Verstappen Lap Times - Abu Dhabi 2024")
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# for lap in pit_in_laps['LapNumber']:
#     plt.axvline(x=lap, color='r', linestyle='--', alpha=0.5)
# for lap in pit_out_laps['LapNumber']:
#     plt.axvline(x=lap, color='g', linestyle='--', alpha=0.5)
# for compound in compounds:
#     stint = laps[laps['Compound'] == compound]
#     plt.plot(stint['LapNumber'], stint['LapTimeSeconds'], 'o-', label=compound)
# plt.legend(title='Tire Compound')
# plt.show()




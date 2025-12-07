import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer


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

# print(laps.columns)


# Marking the pit stop laps/spikes in the graph
pit_in_laps = laps[laps['PitInTime'].notna()]
pit_out_laps = laps[laps['PitOutTime'].notna()]

# Unique tire compounds used
compounds = laps['Compound'].unique()

features = laps[["LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedFL", "TrackStatus"]]
print(features.head())
labels = laps['LapTimeSeconds']

# One-hot encoding - converting cetegorical features into numerical format
features = pd.get_dummies(features, columns=["Compound", "TrackStatus"])

# Splitting the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Standardizing the features
ct = ColumnTransformer([
    ('scale', StandardScaler(), ["LapNumber", "TyreLife", "Sector1Time", "Sector2Time", "Sector3Time", "SpeedFL"])
], remainder='passthrough')

features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)




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




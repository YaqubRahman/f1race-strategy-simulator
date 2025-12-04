import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


print(fastf1.__version__)
fastf1.Cache.enable_cache('cache')  
session = fastf1.get_session(2024, 'Abu Dhabi', 'R') 
session.load()


laps = session.laps.pick_driver('VER') 
print(laps.head())

laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

# Lap times in seconds
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Marking the pit stop laps/spikes in the graph
pit_laps = laps[laps['PitOutTime'].notna()]

plt.figure(figsize=(10,6))
plt.plot(laps['LapNumber'], laps['LapTimeSeconds'], marker='o')
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (s)")
plt.title("Verstappen Lap Times - Abu Dhabi 2024")
plt.grid(which='both', linestyle='--', linewidth=0.5)
for lap in pit_laps['LapNumber']:
    plt.axvline(x=lap, color='r', linestyle='--', alpha=0.5)
plt.show()



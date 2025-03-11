import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic data: Latitude, Longitude, and Risk Level
np.random.seed(42)
n_samples = 500
latitude = np.random.uniform(10, 50, n_samples)   # Example latitudes
longitude = np.random.uniform(-130, -60, n_samples)  # Example longitudes
seismic_activity = np.random.uniform(0, 1, n_samples)  # Random seismic activity level
fault_line_proximity = np.random.uniform(0, 1, n_samples)  # Distance to fault lines
soil_instability = np.random.uniform(0, 1, n_samples)  # Soil type factor

# Assign risk levels: 0 (Low), 1 (Moderate), 2 (High)
risk_labels = np.where(seismic_activity + fault_line_proximity + soil_instability > 2, 2,
                       np.where(seismic_activity + fault_line_proximity + soil_instability > 1, 1, 0))

# Create a DataFrame
df = pd.DataFrame({'Latitude': latitude, 'Longitude': longitude,
                   'SeismicActivity': seismic_activity,
                   'FaultLineProximity': fault_line_proximity,
                   'SoilInstability': soil_instability,
                   'RiskLevel': risk_labels})

# Split into training and testing data
X = df[['Latitude', 'Longitude', 'SeismicActivity', 'FaultLineProximity', 'SoilInstability']]
y = df['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generate a grid for mapping
grid_lat = np.linspace(10, 50, 100)
grid_lon = np.linspace(-130, -60, 100)
grid_latlon = np.array([[lat, lon] for lat in grid_lat for lon in grid_lon])

# Predict risk levels for the grid
grid_features = pd.DataFrame({'Latitude': grid_latlon[:, 0], 'Longitude': grid_latlon[:, 1],
                              'SeismicActivity': np.random.uniform(0, 1, len(grid_latlon)),
                              'FaultLineProximity': np.random.uniform(0, 1, len(grid_latlon)),
                              'SoilInstability': np.random.uniform(0, 1, len(grid_latlon))})
grid_risk = rf_model.predict(grid_features)

# Convert to GeoDataFrame for mapping
gdf = gpd.GeoDataFrame(grid_features, geometry=gpd.points_from_xy(grid_features.Longitude, grid_features.Latitude))
gdf['RiskLevel'] = grid_risk

# Plot the Risk Zones
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(column='RiskLevel', cmap='coolwarm', legend=True, ax=ax, markersize=10)
plt.title("Predicted Earthquake Risk Zones")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


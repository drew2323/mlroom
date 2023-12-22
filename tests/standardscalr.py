import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([
    [32.10, 2343222, 50, 0.5],  # Example observation
    [33.20, 2500000, 60, -0.3], # Another observation
    [31.50, 2000000, 40, 0.8]   # Yet another observation
])

# Creating the StandardScaler instance
scaler = StandardScaler()

# Fitting the scaler to the data and transforming the data
scaled_data = scaler.fit_transform(data)

original_data = data

print("original data", original_data)
print("scaled data", scaled_data)

print(scaler.scale_)
print(scaler.var_)
print("pocet featur:", scaler.n_features_in_)
#print(scaler.feature_names_in_)
print("pocet samplu:", scaler.n_samples_seen_)



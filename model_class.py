from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import pickle

def load_and_prepare_data(data_path):
    data = []
    traffic_volume_map = {0: 'Light', 1: 'Moderate', 2: 'Heavy'}
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Traffic_Volume(target)'] = traffic_volume_map[int(row['Traffic_Volume(target)'])]
            data.append(row)
    return data

def encode_features(data, feature_names):
    feature_values = {feature: set() for feature in feature_names}
    for row in data:
        for feature in feature_names:
            feature_values[feature].add(row[feature])
    
    feature_map = {feature: {val: idx for idx, val in enumerate(sorted(values))} for feature, values in feature_values.items()}
    encoded_data = []
    for row in data:
        row_encoded = []
        for feature in feature_names:
            row_encoded.append(feature_map[feature][row[feature]])
        encoded_data.append(row_encoded)
    
    return encoded_data, feature_map

data_path = '/Users/shahad/Desktop/ML lab/traffic/processed_data.csv'
data = load_and_prepare_data(data_path)
features = ['Age', 'Occupation', 'Residence_Area', 'Movement_Frequency', 'Active_Times', 'Primary_Transport_Mode', 'Vehicle_Type']
y = [row['Traffic_Volume(target)'] for row in data]
X, feature_map = encode_features(data, features)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('feature_map.pkl', 'wb') as f:
    pickle.dump(feature_map, f)

print("Training completed and model saved.")

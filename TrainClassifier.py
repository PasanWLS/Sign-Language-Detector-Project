import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Inspect data to ensure uniformity
data = data_dict['data']
labels = data_dict['labels']

# Find the maximum length of inner lists (if they are lists)
max_length = max(len(item) for item in data)

# Pad or truncate data to ensure uniform shape
data = np.array([np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item[:max_length] for item in data])

# Convert labels to numpy array
labels = np.asarray(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

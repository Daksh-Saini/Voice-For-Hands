# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert to numpy arrays
data = np.array(data_dict['data'], dtype=object)
labels = np.array(data_dict['labels'])

# Make sure every sample has 42 features
cleaned_data = []
for sample in data:
    if len(sample) == 42:
        cleaned_data.append(sample)
    elif len(sample) < 42:
        cleaned_data.append(sample + [0.0] * (42 - len(sample)))
    else:
        cleaned_data.append(sample[:42])

cleaned_data = np.array(cleaned_data, dtype=float)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    cleaned_data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score*100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

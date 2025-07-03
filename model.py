import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Features & labels
X = df.iloc[:, :-1]
y = df['target']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as iris_model.pkl")

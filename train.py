import os
import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Create data directory
os.makedirs("data", exist_ok=True)

# Load dataset
dataset = load_breast_cancer()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

# Save dataset
df.to_csv("data/breast_cancer.csv", index=False)

X = dataset.data
y = dataset.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN Model
model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

# Save ANN model
model.save("data/ann_model.h5")

# Save scaler
with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


print("Model saved successfully!")
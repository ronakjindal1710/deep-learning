import numpy as np
from sklearn.datasets import make_moons

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD


# Make Moons dataset
X, y = make_moons(
    n_samples=400,
    noise=0.20,
    random_state=1
)

y = y.reshape(-1, 1).astype(float)

# Standardize the input
X = (X - X.mean(0)) / X.std(0)


# Model
model = Sequential([
    Input(shape=(2,)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(learning_rate=0.5),
    metrics=['accuracy']
)


# Batch Gradient Descent
# Use the whole dataset as one batch
model.fit(
    X,
    y,
    epochs=200,
    batch_size=len(X),
    verbose=0
)

loss, accuracy = model.evaluate(X, y, verbose=0)
print("Batch Gradient Descent accuracy:", accuracy)


# Stochastic / Mini-Batch Gradient Descent
# Small batch size -> many updates per epoch
model.fit(
    X,
    y,
    epochs=200,
    batch_size=16,
    verbose=0
)

loss, accuracy = model.evaluate(X, y, verbose=0)
print("Mini-Batch Gradient Descent accuracy:", accuracy)
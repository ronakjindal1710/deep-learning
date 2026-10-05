import numpy as np

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_derivative(a): return a * (1 - a) # a is the activation

inputs = np.array([[0,0],[0,1],[1,0],[1,1]]) # XOR truth tableexpected_output = np.array([[0], [1], [1], [0]])
expected_output = np.array([[0], [1], [1], [0]])
inN, hidN, outN = 2, 2, 1 # 2-2-1 network
hidden_weights = np.random.uniform(size=(inN, hidN))
hidden_bias = np.random.uniform(size=(1, hidN))
output_weights = np.random.uniform(size=(hidN, outN))
output_bias = np.random.uniform(size=(1, outN))
lr = 0.1

for epoch in range(10000):
    # ---- forward ----
    h_out = sigmoid(np.dot(inputs, hidden_weights) + hidden_bias)
    predicted_output = sigmoid(np.dot(h_out, output_weights) + output_bias)

    # ---- backward ----
    error = expected_output - predicted_output
    d_out = error * sigmoid_derivative(predicted_output) # delta_o
    d_hid = d_out.dot(output_weights.T) * sigmoid_derivative(h_out) # delta_h
    
     # ---- update ----
    output_weights += h_out.T.dot(d_out) * lr
    output_bias += np.sum(d_out, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hid) * lr
    hidden_bias += np.sum(d_hid, axis=0, keepdims=True) * lr
    
print(np.round(predicted_output, 3))
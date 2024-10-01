import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def forward(inputs, weights,bias):
    sum = 0
    for i in range(len(inputs)):
        sum += inputs[i] * weights[i]
    sum += bias
    return sigmoid(sum)

def backward(inp , w , lr, target ,op):
    for i in range(len(w)):
        delw = lr * (target - op) * inp[i]
        w[i] += delw

x = np.array([0,0,1,1])
y = np.array([0,1,0,1])
weights = np.array([0.2,0.1])
ran = np.random.default_rng()
for i in range(len(weights)):
    weights[i] = ran.random()
print("init weights")
print(weights)
print("changing weights")
print(forward([0,1],weights,0))
for iterations in range(10000):
    for i in range(len(x)):
        out = forward([x[i],y[i]],weights,0)
        target = x[i] & y[i]
        backward([x[i],y[i]],weights,0.01,target,out)
        print(weights)
print(forward([0,1],weights,0))

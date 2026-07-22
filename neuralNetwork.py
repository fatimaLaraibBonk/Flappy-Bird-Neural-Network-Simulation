import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import random 
from collections import deque
#Inputs 
#inputs rperesent the current state 

#Hidden Layers

#Output a vector of Q values representing the expected rewards for each action 
#in our case the actions are 2, so vector size n=2

class NeuralNetwork:
    
    def __init__(self, numOfInputs, numOfOutputs, hiddenLayers, numOfNeuronsPerHiddenLayer):
        #basically define the skeleton of the neural network 
        self.numOfInputs=numOfInputs
        self.numOfOutputs=numOfOutputs
        self.hiddenLayers=hiddenLayers
        self.numOfNeuronsPerHiddenLayer=numOfNeuronsPerHiddenLayer
        self.numOfTotalLayers=hiddenLayers+2 
        self.weights=[]
        self.biases=[]
        self.layer_sizes = (
            [numOfInputs]
            + numOfNeuronsPerHiddenLayer
            + [numOfOutputs]
        )
        for i in range(self.numOfTotalLayers-1):
            self.biases.append(
                np.random.randn(self.layer_sizes[i+1], 1)*0.01
            )
        self.cache={"activations":[], "z_values":[]}
        self.epsilon=1
        self.epsilon_min=0.05
        self.epsilon_decay=0.995
        self.memory=deque(maxlen=50000)
        self.batch_size=32
        self.min_replay_size=500
        #where 2 is for the input and output lauer
        #however since the weights are always totalLayers-1 you will often see totalLayers-1 being presented as totalLayers 
    
    def initialize_weights(self):
        #following row column notation 
        #example so if i have two (2) hidden layers 4 inputs 5 neurons for hidden layer 1 8 neruons for hidden layer 2 and then 2 for poutout id have these weight matrives 5x4 8x5 2x8
        for i in range(self.numOfTotalLayers-1):
            if i==0:
                #dealing with the input layer
                fan_in=self.numOfInputs
                weights=np.random.randn(self.numOfNeuronsPerHiddenLayer[0], self.numOfInputs)* np.sqrt(2.0 / fan_in)
            elif i==self.numOfTotalLayers-2:
                #dealing with the output layer
                fan_in=self.numOfNeuronsPerHiddenLayer[-1]
                weights=np.random.randn(self.numOfOutputs, self.numOfNeuronsPerHiddenLayer[-1])* np.sqrt(2.0 / fan_in)
            else:
                #dealing with inner layers 
                fan_in=self.numOfNeuronsPerHiddenLayer[i-1]
                weights=np.random.randn(self.numOfNeuronsPerHiddenLayer[i], self.numOfNeuronsPerHiddenLayer[i-1])* np.sqrt(2.0 / fan_in)
            self.weights.append(weights)
        pass

    def calculation(self, state):
        activations=[state]
        z_values=[]
        a=state
        for i in range(self.numOfTotalLayers-1):
            if i==0:
                #multply inputs with weights
                result=self.weights[0]@state + self.biases[0]
                #applying activation function 
                a=self.activation(result)
                z_values.append(result)
                activations.append(a)
            elif i==self.numOfTotalLayers-2:
                #finall multiplication to get output 
                result=self.weights[-1]@a+self.biases[-1]
                z_values.append(result)
                return result, activations, z_values 
            else:
                #hidden layer multiplication 
                result=self.weights[i]@a+self.biases[i]
                a=self.activation(result)
                z_values.append(result)
                activations.append(a)
        pass

    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def activation(self, x):
        #reLu 
        return np.maximum(0,x)
    
    def action(self, current_state):
        if random.random() < self.epsilon:
            return random.randint(0,1)
        q_values, _, _=self.calculation(current_state)
        return np.argmax(q_values)
    
    def _train_on_one(self, state, action, reward, next_state, done):
        gamma = 0.99
        learning_rate = 0.001

        # 1. Forward pass for current state (keeps its own activations/z_values)
        current_q, state_activations, state_z_values = self.calculation(state)

        # 2. Forward pass for next state (isolated, won't overwrite state variables)
        next_q, _, _ = self.calculation(next_state)

        # 3. Compute target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + gamma * np.max(next_q)

        target_q = np.clip(target_q, -50, 50)
        
        target = current_q.copy()
        target[action] = target_q
        
        # Loss gradient wrt output: dL/d(Q)
        delta = current_q - target

        # 4. Backpropagation using state_activations and state_z_values directly
        for layer in reversed(range(len(self.weights))):
            a_prev = state_activations[layer]
            
            dW = delta @ a_prev.T
            db = delta

            # Gradient clipping
            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            # Backpropagate error gradient to lower layer before updating weights
            if layer > 0:
                delta = self.weights[layer].T @ delta
                z_prev = state_z_values[layer - 1]
                delta = delta * self.relu_derivative(z_prev)

            # Apply weight and bias updates
            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db
        

    def backpropogation(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.min_replay_size:
            self._train_on_one(state, action, reward, next_state, done)
        else:
            batch = random.sample(self.memory, self.batch_size)
            for s, a, r, ns, d in batch:
                self._train_on_one(s, a, r, ns, d)

    def visualise_error(self):
        pass

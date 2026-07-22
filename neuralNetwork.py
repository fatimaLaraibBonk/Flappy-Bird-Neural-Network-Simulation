import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import random 
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

        #where 2 is for the input and output lauer
        #however since the weights are always totalLayers-1 you will often see totalLayers-1 being presented as totalLayers 
    
    def initialize_weights(self):
        #following row column notation 
        #example so if i have two (2) hidden layers 4 inputs 5 neurons for hidden layer 1 8 neruons for hidden layer 2 and then 2 for poutout id have these weight matrives 5x4 8x5 2x8
        for i in range(self.numOfTotalLayers-1):
            if i==0:
                #dealing with the input layer
                weights=np.random.randn(self.numOfNeuronsPerHiddenLayer[0], self.numOfInputs)*0.01
            elif i==self.numOfTotalLayers-2:
                #dealing with the output layer
                weights=np.random.randn(self.numOfOutputs, self.numOfNeuronsPerHiddenLayer[-1])*0.01
            else:
                #dealing with inner layers 
                weights=np.random.randn(self.numOfNeuronsPerHiddenLayer[i], self.numOfNeuronsPerHiddenLayer[i-1])*0.01
            self.weights.append(weights)
        pass

    def calculation(self, state):
        self.cache["activations"]=[state]
        self.cache["z_values"]=[]
        a=state
        for i in range(self.numOfTotalLayers-1):
            if i==0:
                #multply inputs with weights
                result=self.weights[0]@state + self.biases[0]
                #applying activation function 
                a=self.activation(result)
                self.cache["z_values"].append(result)
                self.cache["activations"].append(a)
            elif i==self.numOfTotalLayers-2:
                #finall multiplication to get output 
                result=self.weights[-1]@a+self.biases[-1]
                self.cache["z_values"].append(result)
                return result #Q values 
            else:
                #hidden layer multiplication 
                result=self.weights[i]@a+self.biases[i]
                a=self.activation(result)
                self.cache["z_values"].append(result)
                self.cache["activations"].append(a)
        pass

    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def activation(self, x):
        #reLu 
        return np.maximum(0,x)
    
    def action(self, current_state):
        if random.random() < 0.1:
            return random.randint(0,1)
        action=np.argmax(self.calculation(current_state))
        return action 

    def backpropogation(self, state, action, reward, next_state, done):
        gamma=0.99
        learning_rate=0.001
        #here we will be using the bellmans equation 
        #So first we need to calculate the Q value for our state
        #Then we compare it with the outputs of the forward propogation 
        #then we apply our lost function 
        #finally we update the weights
        current_q=self.calculation(state)
        state_activations=self.cache["activations"].copy()
        state_z_values=self.cache["z_values"].copy()
        next_q=self.calculation(next_state) 
        self.cache["activations"] = state_activations
        self.cache["z_values"] = state_z_values
        #Qtarget=reward+γ(next_q)
        #where γ is the discount factor
        if done:
            target_q=reward
        else:
            target_q=reward+gamma*np.max(next_q)
        target=current_q.copy()
        target[action]=target_q
        predicted_q=current_q[action]
        loss=(target_q-predicted_q)**2
        error=target_q-predicted_q
        delta=current_q-target
        #chain rule for finding the derivatives
        #L/Output into Output/Z into Z/W 
        for layer in reversed(range(len(self.weights))):
            a_prev=self.cache["activations"][layer]
            dW=delta@a_prev.T
            db=delta
            self.weights[layer]-=learning_rate * dW
            self.biases[layer]-=learning_rate * db
            if layer>0:
                delta=self.weights[layer].T@delta
                z_prev=self.cache["z_values"][layer - 1]
                delta=delta * self.relu_derivative(z_prev)
        #now we need to find the derivatives


        #and using those derivatives we can update our weights 

        pass

    def visualise_error(self):
        pass

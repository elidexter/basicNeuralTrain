import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights=2*np.random.random((3,1))-1
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivated(self,x):
        return x*(1-x)
    
    def train(self,training_inputs,training_outputs,training_interations):
        for iteration in range(training_interations):
            output=self.think(training_inputs)
            error=training_outputs-output
            adjustemnts=np.dot(training_inputs.T,error*self.sigmoid_derivated(output))
            self.synaptic_weights+=adjustemnts
    def think(self,inputs):
        inputs=inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))
        return output

if __name__=="__main__":
    neuralNetwork=NeuralNetwork()    
    training_inputs=np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
    training_outputs=np.array([[0,1,1,0]]).T
    neuralNetwork.train(training_inputs,training_outputs,10000)
    print('Synaptic after training:')
    print(neuralNetwork.synaptic_weights)
    A=str(input("Input A"))
    B=str(input("Input B"))
    C=str(input("Input C"))
    print('Situation',A,B,C)
    value=neuralNetwork.think(np.array([A,B,C]))
    print('Result ',value)
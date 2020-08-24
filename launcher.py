import NeuralNetwork
import numpy as np

if __name__=="__main__":
    neuralNetwork=NeuralNetwork.NeuralNetwork()    
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
import math as m
import numpy as np
import pandas as pd

np.random.seed(2)

class NeuralNetwork:
    # Class constructor method
    def __init__(self, epochs, inputs, hidden_nodes, rho, alpha, bd, training_data, testing_data, min_max):
        # Instance variables initalised
        self.inputs = inputs
        self.hidden_nodes = hidden_nodes
        self.rho = rho
        self.training_data = training_data.copy()
        self.testing_data = testing_data.copy()
        self.min_max = min_max
        self.epochs = epochs
        self.alpha = alpha
        self.use_bd = bd
        # Class variables
        self.node_activations = [] 
        self.error_array = []
        self.hidden = NeuralNetwork.get_hidden(self)
        self.output = NeuralNetwork.get_output(self)
        self.error = 0
        self.run_algorithm()

    def run_algorithm(self):
        count = 0
        error_old = 10000
        bd_hidden = self.hidden
        bd_output = self.output
        for i in range(self.epochs): # Number of epochs to run
            self.backpropagation() # Run the backpropagation algorithm
            mse = (self.destandard(self.error,self.min_max[0],self.min_max[1]))/(len(self.training_data.index)) # Calculates MSE
            if(self.use_bd==True):
                if count == 25: # BOLD DRIVER: EVERY 50 epochs change the learning rate
                    if error_old>mse: # BOLD DRIVER: If error has decreased
                        self.rho = self.rho * 1.05
                        if self.rho>0.5:
                            self.rho = 0.5
                        bd_hidden = self.hidden
                        bd_output = self.output
                        print('Previous: %f, Current: %f' % (error_old,mse))
                        error_old = mse
                    if error_old<mse : # BOLD DRIVER: If error has increased
                        self.hidden = bd_hidden
                        self.output = bd_output
                        self.rho = self.rho * 0.7
                        if self.rho<0.01:
                            self.rho = 0.01
                        error_old = 10000
                    count=0
            print('epoch: %d, error: %f, rho: %f' % (i,mse,self.rho))
            self.error = 0
            count += 1
        self.prediction()

    # Set initial weights from input to the hidden and set hidden node biases
    def get_hidden(self):
        hidden_layer = [{'weights':[(round(np.random.uniform((-2/self.inputs),(2/self.inputs)),4)) for i in range(self.hidden_nodes * self.inputs)],
        'biases':[(round(np.random.uniform((-2/self.inputs),(2/self.inputs)),4)) for i in range(self.hidden_nodes)]}]
        return hidden_layer

    # Set initial weights from hidden to the ouput and set output node bias
    def get_output(self):
        output_layer = [{'weights':[(round(np.random.uniform((-2/self.inputs),(2/self.inputs)),4)) for i in range(self.hidden_nodes)],
        'biases':[(round(np.random.uniform((-2/self.inputs),(2/self.inputs)),4))]}]
        return output_layer

    def backpropagation(self): # START OF BACKPROPAGATION ALGORITHM
        for row in self.training_data.itertuples(index=False, name='Input'):
            expected = row[-1]
            self.forward_pass(row) # Starts a forward pass on the current row
            self.back_pass(row) # Starts a back pass
            self.error += ((self.node_activations[-1]-expected)**2) # Adds error to be used for MSE
            self.node_activations.clear() # Clears previous node activations
            self.error_array.clear() # Clear node errrors

    def destandard(self,back_out, min, max): # Destandardise the values
            output = (((back_out-0.1)/0.8)*(max-min))+min 
            return output
        
    def forward_pass(self,row): # Forward pass
        for i in range(self.hidden_nodes): # For each hidden node
            node_weight_sum = 0
            for j in range(self.inputs): # For each input
                answer = self.hidden[0]["weights"][j+(i*(self.inputs))] * row[j]
                node_weight_sum += answer
            node_weight_sum += self.hidden[0]["biases"][i] # Add hidden node bias
            # hidden_output = 1.00/(1.00 + m.exp(-node_weight_sum)) # Use sigmoid transfer function for hidden layer
            hidden_output = np.tanh(node_weight_sum) # Use tanh transfer function for hidden layer
            self.node_activations.append(hidden_output)
        output_weight_sum = 0
        for j in range(self.hidden_nodes):
            output_weight_sum += self.output[0]["weights"][j] * self.node_activations[j]
        output_weight_sum += self.output[0]["biases"][0]
        # output_output = 1.00/(1.00 + m.exp(-output_weight_sum)) # Use sigmoid transfer function for hidden layer
        output_output = np.tanh(output_weight_sum) # Use tanh transfer function for hidden layer
        self.node_activations.append(output_output) # Append to node activations

    def prediction(self):
        f = open(f'{self.hidden_nodes}_{self.rho}_{self.alpha}_{self.use_bd}.csv', "x") # Create text file with config as the name
        error = 0
        for row in self.testing_data.itertuples(index=False, name='Input'):
            self.forward_pass(row) # Perfrom a forward pass
            error += (((row[-1] - self.node_activations[-1]))**2) 
            f.write('%4f, %4f \n' % (self.destandard(row[-1],self.min_max[2],self.min_max[3]),self.destandard(self.node_activations[-1],self.min_max[2],self.min_max[3]))) # Write to text file
            self.node_activations.clear()
        f.close()
        
    def back_pass(self, row): # Back pass
        self.output_to_hidden(row) # Calculates error on the node
        self.update_input_weights(row) 
        self.update_output_weights()
        self.update_biases()

    def output_to_hidden(self,row):
        for i in range((len(self.node_activations)-1),-1,-1):
            output = self.node_activations[i]
            if i==(len(self.node_activations)-1):
                expected = row[-1]
                output_error = (expected-output) * self.t_derivative(output)
                self.error_array.append(output_error)
            else:
                error = (self.output[0]["weights"][i] * output_error) * self.t_derivative(output)
                self.error_array.insert(0,error)
    
    def update_input_weights(self, row):
        for i in range(self.hidden_nodes): # For each hidden node
            for j in range(self.inputs):
                weight_new = self.hidden[0]['weights'][j+(i*(self.inputs))] + (self.rho * self.error_array[i] * row[j])
                weight_new_moment = weight_new + (((weight_new)-(self.hidden[0]['weights'][j+(i*(self.inputs))])) * self.alpha) # Adds momentum
                self.hidden[0]['weights'][j+(i*(self.inputs))] = weight_new_moment
        
    def update_output_weights(self):
        for i in range(self.hidden_nodes):
            weight_new = self.output[0]['weights'][i] + (self.rho * self.error_array[-1] * self.output[0]['biases'][0])
            weight_new_moment = weight_new + (((weight_new)-(self.output[0]['weights'][i])) * self.alpha) # Adds momentum
            self.output[0]['weights'][i] = weight_new_moment

    def update_biases(self): 
        for i in range(self.hidden_nodes):
            bias_new = self.hidden[0]['biases'][i] + (self.rho * self.error_array[i] * 1)
            self.hidden[0]['biases'][i] = bias_new
        output_bias = self.output[0]['biases'][0] + (self.rho * self.error_array[-1] * 1)
        self.output[0]['biases'][0] = output_bias

    def t_derivative(self, output):
	    # return output * (1.0 - output) # sigmoid
        return (1-(np.tanh(output))**2) # tanh derivative

# END OF CLASS
def import_data(file_name): # Imports full dataset
    dataset = pd.read_csv(file_name)
    return dataset

def g_training(dataset,test_size): # Returns dataset without training data
    train_data = dataset.copy()
    df_train = train_data[:-test_size]
    return df_train

def g_testing(dataset,test_size): # Returns training data
    test_data = dataset.copy()
    df_test = test_data[(-test_size-1):]
    return df_test

def standardize(data): # Standardises the dataset between 0.1 and 0.9
    for header in data.columns:
        if header != "Date":
            min_value = data[header].min()
            max_value = data[header].max()
            data[header] = np.round(((0.8*((data[header]-min_value)/(max_value-min_value)))+0.1),4)
    return data

def data_import(file_name):
    dataset = import_data(file_name)
    return dataset

def get_min_max(data, data2): # Gets min/max from training data and testing data
    return [data.iloc[:,-1].min(),data.iloc[:,-1].max(),data2.iloc[:,-1].min(),data2.iloc[:,-1].max()]

def main():
    print("Starting program")
    dataset = data_import('data.csv')
    t_size = 365
    training = standardize(g_training(dataset,t_size)) # Slices then standardises the data
    training.drop(columns=training.columns[0], axis=1, inplace=True) 
    testing = standardize(g_testing(dataset,t_size)) # Slices then standardises the data
    testing.drop(columns=testing.columns[0], axis=1, inplace=True) # Removes data
    min_max = get_min_max(g_training(dataset,t_size),g_testing(dataset,t_size)) 
    instance2 = NeuralNetwork(1000,8,16,0.01,0.9,False,training,testing,min_max) # Creates an instance of the class
    
if __name__ == "__main__":
    main()
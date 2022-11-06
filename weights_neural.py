# class_set = [[0,0,1],[1,0,1],[1,1,0],[1,0,0],[0,0,0],[0,1,1]]
# data_set = [0,0,1,0,0,1]
# weights = [0,0,0]

class Weigths:
    
    def __init__(self,class_set,data_set):
        self.class_set = class_set
        self.data_set = data_set
        self.weights = [0.0 for _ in class_set[0]]

        self.F_train(class_set,data_set,3)
        # self.F_predict(INPUTS)
        # print("weights:",self.weights)

    
    def math_exp(self,x):
        return 2.718281828459045**x
    
    def F_sum_of_weighted_input(self,neuron_inputs):
        sum_of_weighted_inputs = 0
        for index_num in range(len(neuron_inputs)):
            sum_of_weighted_inputs += self.weights[index_num] * neuron_inputs[index_num]
        return sum_of_weighted_inputs
    
    def F_predict(self,neuron_inputs):
        sum_of_weighted_inputs =  self.F_sum_of_weighted_input(neuron_inputs)
        neuron_input = self.F_activation(sum_of_weighted_inputs)
        return neuron_input
    
    def F_adjust_weight(self,neuron_input, error_in_output,predicted_output):
        sigmoid_gradient = predicted_output * (1-predicted_output)
        adjust_weight = neuron_input * error_in_output * sigmoid_gradient
        return adjust_weight
    
    def F_train(self,training_set_examlps,result_set,number_of_iterations):
        for iteration in range(number_of_iterations):
            for training_num in range(len(training_set_examlps)):
                predict_output = self.F_predict(training_set_examlps[training_num])
                error_in_output = self.F_error_cost(result_set[training_num],predict_output)
                for index_num in range(len(self.weights)):
                    neuron_input = training_set_examlps[training_num][index_num]
                    self.weights[index_num] += self.F_adjust_weight(neuron_input, error_in_output, predict_output) 
    
    def F_activation(self,sum_of_weighted_inputs):
        return 1/(1+self.math_exp(-sum_of_weighted_inputs))
    
    def F_error_cost(self,output, predicted):
        error_in_output = output - predicted
        return error_in_output


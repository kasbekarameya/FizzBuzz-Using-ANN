# FizzBuzz-Using-ANN
Recently, Andrej Karpathy, Director of AI at Tesla, proposed the idea of a new link of software. He called it Software 2.0 and proposed that Artificial Neural Networks can be used to develop it. This concept is completely different from what methodology we have used to develop any software till date, which can be described as Software 1.0. The goal of this project is to compare the performance based on accuracy of Software 2.0 with the traditional Software 1.0, using the Fizz Buzz problem.

# Software 1.0
Software 1.0 is what has been the representation of all the current software definitions. It can be described as a method of explicitly writing code in which the programmer has to think of each & every condition that may or may not occur while the program is in execution. 
Almost all the software used and developed in the IT industry today is based on the concept of Software 1.0. That being said the new & emerging concept of Software 2.0 is here to challenge the Software 1.0 industry space. Let us now look at the concept put forth by Software 2.0  

# Software 2.0
Due to the advent of the concepts of Neural Network, the concept of Software 2.0 was introduced. Compared to Software 1.0, Software 2.0 can be described as an approach used to model a software wherein there is less programmer involvement in the actual working of the software.
Unlike Software 1.0, wherein all the conditions & decisions are hardcoded into the system, Software 2.0 depends more on the probability and weights of individual decisions taken by the software. Although it has not become a well-recognized concept yet, software 2.0 has been increasing in understanding & popularity within the IT industry.  

# The Fizz Buzz Problem

Fizz Buzz is a popular children’s game played in schools in the UK to help them learn division. Recently due to its logic and structure, Fizz Buzz is being used as a popular coding question in IT based job interviews. 
The concept of Fizz Buzz is that the programmer has to print a set of numbers from 1 to n, such that if the number is divisible by 3 then the output should be ‘Fizz’, if by 5 then the output should be ‘Buzz’, if it is divisible by both 3 & 5 then it should be ‘Fizz Buzz’, or else we need to print the number itself.
In this project, we are trying to implement, the Fizz Buzz Problem using both the Software 1.0 & Software 2.0 approaches, in order to compare the performance & other metrics between the two Software models.    

# Understanding Neural Networks
In the past few years, there has emerged a new field of study in Machine Learning that is being used widely throughout the IT Industry and it is popularly known as Artificial Neural Networks. Artificial Neural Networks, also abbreviated as ANN, are build based on the concept of Human Brain. Like the human brain, the ANN is comprised of n number of neurons that work together to model complex pattern & prediction problems.
For any ANN architecture, there is one input layer, one or more than one hidden layer & one output layer. The multiple hidden layers are used to determine distinctive patterns in the data taken as input & also to increase the accuracy of the model used . 
Weights ‘W’ is a concept that is used to determine the importance of each neuron in the network. Hence more the weight associated with a neuron, more important it is for generating the required output. In order to increase the efficiency of the neural network, we need to adjust weights assigned to every edge connecting two neurons in the network.

# Software 1.0 Vs Software 2.0
Now that we understand the concept of Artificial Neural Network & basics of Python Programming Language, we need compare both the Software 1.0 & Software 2.0 approaches using various metrics.

## Accuracy in Software 1.0
The code used to implement Software 1.0 is written in Python. It uses the basic methodology of implementing a program wherein a software programmer hardcodes all the possible conditions that can occur during the execution of the program. 
It is because of this property of the Software 1.0, until and unless the execution fails or the logic in implementing the program fails we will always obtain output and in turn 100% accuracy after executing the program.

## Accuracy in Software 2.0
Unlike Software 1.0, Software 2.0 follows the principle of machine learning wherein we provide data to the model & predefine the output that has to be achieved by it. Then based on previous data the model itself has to increment towards achieving the required output. 
In order to achieve the required output, each model has a predefined set of values known as Hyper Parameters. Hyper Parameters are variable in an Artificial Neural Network that decide the structure of the network & also determine how good the network works. Hence, in order to increase output accuracy of a Neural Network, we have to adjust the values of these parameters. Now let’s try to improve the accuracy by varying the Hyper Parameters. Here we are evaluating the performance of the Neural Network based on following four parameters;
*Training Accuracy: This parameter is the accuracy of the model to determine the correct label of the data based on the training dataset.
*Validation Accuracy: This parameter is the accuracy of the model to determine the correct label of the data based on the validation dataset.

## Number of Layers
A layer in a Neural Network is a combination of neurons working together to provide a determined output. The accuracy of a model is dependent on how many number of hidden layers are present in a network [6].
Based on the variation performed in the parameter it is safe to say that number of hidden layers in a network is directly proportional to the accuracy & loss of the model. Hence we have to add more hidden layers in the neural network until the accuracy & loss scores do not improve anymore [6].
In this particular neural network model, after we increase the number of layers parameter to Number of Layers = 4, there is a lot of variations in the validation accuracy graph, but we get the maximum possible Testing Accuracy = 0.97, Validation Accuracy = 0.88.
    
Figure 2: Validation & Test Accuracy for a 4 layer Neural Network

## Batch Size of the Model
Batch Size can be defined as the number of sub samples of the whole input dataset, given as a input to a network at a time [6]. Batch Size of a model is inversely proportional to the amount of memory used by the model to compute the output. This means that larger the batch size, lesser number of times the model has to run to provide the output and vice versa.
In this particular neural network model, when the Batch Size = 32, the validation accuracy graph shows a lot of variation, but the overall accuracy parameters i.e. Testing Accuracy = 0.88, Validation Accuracy = 0.81 demonstrate almost above average values.
  
Figure 3: Validation & Test Accuracy for a Neural Network with Batch Size = 32

## Dropout Rate
In larger neural networks there is a chance of overfitting the model to the input data. Hence there a parameter called as Dropout rate, which is a part of a regularization process that is used to avoid overfitting the data. Avoiding overfitting will in turn increase the validation accuracy [6]. 
Generally, we assign 20% to 50% dropout rate, which means that 20% to 50% of neurons are dropped from the network [6]. Too high value of this parameter will cause under learning, but too low value will have minimal effect on the output.
In this particular neural network model, after setting the value of the Dropout Rate = 0.7 or 70%, the validation accuracy graph shows rapid growth after third batch of input and the Testing Accuracy = 0.88 & Validation Accuracy = 0.81, which are more than average values.
  
Figure 4: Validation & Test Accuracy for a Neural Network with a 70% Dropout Rate 

## Activation Function
Activation Functions are used by Neural Networks as an approximation function from one layer to another. Activation Functions work to introduce nonlinearity in the models [6]. 
One of the most used function is the rectifier activation function ‘relu’. Also ‘sigmoid’ function is used in neural networks making binary predictions, whereas ‘softmax’ function is used in neural networks making multi class predictions [6].
In this particular neural network model, value of the activation function = relu is the optimal value of this parameter as it outputs maximum performance based on this parameter. The value for Testing Accuracy is 0.84 & Validation Accuracy is 0.75, which are more than average values.
  
Figure 5: Validation & Test Accuracy using the ‘Relu’ Activation Function

## Optimizer & Learning Rate
Learning Rate can be defined as the rate at which the model reaches to its optimal value. In other words, learning rate is the rate at which the neural network updates it weights & other parameters [6].
Low learning rate will slow down the learning process & increases the training time of the model. Whereas high learning rate will speed up the learning process, but on the other hand it is possible that model may miss its target value. Optimizer is used to determine which learning rate function will be used to evaluate the optimal value. 
In this particular neural network model, value of the optimizer = ‘rmsprop’ & ‘Adam’ are the optimal values of this parameter as it outputs maximum performance based on it. The value for Testing Accuracy is 0.89 & Validation Accuracy is 0.80, which are more than average values.
  
Figure 6: Validation & Test Accuracy using the ‘Adam’ Optimizer

## Number of Epochs
Epoch can be defined as one iteration, in a forward & backward manner, over the entire training dataset by the neural network model.
Too low number of epochs will cause under fitting of the data into the model, whereas too many number of epochs will cause the model to overfit the data provided. We should ideally increase the number of epochs until the validation accuracy starts to decrease, even when test accuracy increases [6]. 
In this particular neural network model, we have set the value of the parameter number of epochs to only 100 epochs. Due to this as shown by the graphs there is not many epochs to train the model and hence it causes under fitting of data. Moreover, we also obtain a very low Testing Accuracy value of 0.54 and a Validation accuracy value of 0.53 respectively.
  
Figure 7: Validation & Test Accuracy with only 100 Epochs
 
Hence, after multiple iterations, I can state that the optimal values for the Hyper Parameters, in order to obtain the maximum probability are:
Number of Layers = 4 (including 2 hidden layers), Batch Size = 128, Dropout Rate= 0.2, Activation Function = relu, Optimizer = rmsprop & Number of Epochs:10000
Using these value, I was able to obtain Test Accuracy = 0.94 & Validation Accuracy = 0.89. Also the graphs generated using this setting are very close to the ideal graphs required.
  
Figure 8: Validation & Test Accuracy with optimal values for hyper parameters

# In Conclusion

The major difference between Software 1.0 & Software 2.0 is that, 1.0 is the code written by a programmer, whereas 2.0 is an attempt to write a code by a model designed to solve a specific problem. Software 2.0 will be replacing Software 1.0, wherever automation is required and possible.
Moreover, by performing this project, I was able to determine a number of advantages Software 2.0 brings to the IT Industry. Some of them are as follows:
*Computationally homogenous design
*Constant running time & memory use
*Module can be combined to code
*Is highly agile & portable
*Can perform some operations even better than humans

In order to complete this project, the process used is as follows:
*Build a program representing Software 1.0 concept using Python
*Build a neural network model representing the concept of Software 2.0 using Python
*Calculate the accuracy of Software 1.0 program
*Vary the Hyper Parameters to obtain maximum possible accuracy from Software 2.0 model

In conclusion, I can state that by performing this project we were able to determine the efficiency and possibilities of tasks that can be accomplished by Software 2.0. I was also able to learn the difference this new Software 2.0 model holds from the programmer’s point of view, compared to the traditional Software 1.0 program.






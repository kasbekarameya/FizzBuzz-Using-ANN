

import pandas as pd

def fizzbuzz(n):
    
    # It is a implementation of the tranditional Software 1.0 methodology. Here, we use if-else condition statements to determine whether to return 'Fizz', 'Buzz' or 'FizzBuzz' or to return the number itself. 
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format


def createInputCSV(start,end,filename):
    
    # Lists are a built-in data sructure offered by Python. They can be used to store an ordered collection of data, which may or may not be homogeneous in nature. We have used lists in this program, because unlike tuples, the values stored in a list are not static in nature. Moreover, list provides a lot of built-in functions such as append(),etc.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Dataframe is a data structure provided by the Pandas Data Analysis library. Dataframes represent the data stored in them in a tabular format. This allows accessing & representing the values from a large dataset very easily.
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# Processing Input and Label Data

def processData(dataset):
    
    # Machine Learning models need data that can be used to work on and provide the required output. Even though data is avaiable in abundance all around us, it is in its raw form i.e. such data contains errors, Not A Number(NAN) and many other elements that are of no use to the ML Model. Hence in order to remove these unwanted elements & convert the data to information, we need to process the data.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    print(processedData)
    
    return processedData, processedLabel


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# Model Definition

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np

input_size = 10
drop_out = 0.2
first_dense_layer_nodes  = 128
second_dense_layer_nodes = 4

def get_model():
    
    # A model is a defination of a neural network that will decide how the network will actually work. The model is a critcal part of any machine learning program because it helps in deciding various factors involved in a machine learning code such as algorithm used, type of dataset used, learning rate of the algorithm and much more.
    # Dense is a type of Neural Network Layer that allows all the node of one layer to be connected to every node of the next layer using weights. On the other hand, an Activation function is a value approximation function used by neural networks to bring linearity to the network model. The reason why we use the dense() function first and then the activation function is that we have to first define the layer using dense() and then approximate its output for the next layer is the activation function.
    # The Sequential model is defined as a model that can be used to add layers to a neural network in a sequential manner. By using the add() method of the Sequentail model object, we can independently add as many layers as we want such that each new layer takes the output of the previous layer as a input. Due to this its ease of use and ability to independently add layers, we are using the Sequential Model.
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Dropout rate can be described as the rate of neurons that will be dropped from the network for every training process. If we run the model for too many epochs, there is a possibility of overfitting the model on the training data. In order to avoid overfitting, we are using this technique called Dropout Rate, which is a part of the Regularization process. Generally 20% -50% dropout rate will be enough for a model.
    model.add(Dropout(drop_out))
    
    model.add(Dense(first_dense_layer_nodes, input_dim=first_dense_layer_nodes))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Softmax is a activation function that assigns probabilites to each classification class in the problem such that their sum much be 1. Softmax is used in models that are working on multi-class problems. As the output layer of this model have four classes i.e. Fizz, Buzz, FizzBuzz & Other, we have to use Softmax activation function in the output layer.
    
    model.summary()
    
    # Categorical_crossentropy loss function can be described as a loss function that unlike Binary crossenntropy function works on multi-class output models. This loss function orders the outputs into various categories, hence the name Categrical Cross Entropy. Moreover unlike Binary CrossEntropy which uses the sigmoid activation function, this loss function uses the softmax activation function. It is because of these reasons that we are using the Categorical CrossEntropy Loss Function. 
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


model = get_model()


validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128 
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

#get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(12,20))


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "ameyakir")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50292574")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')


'''
    Col 1. Number of times pregnant
    Col 2. Plasma glucose concentration a 2 hours in an oral glucose tol
    Col 3. Diastolic blood pressure (mm Hg)
    Col 4. Triceps skin fold thickness (mm)
    Col 5. 2-Hour serum insulin (mu U/ml)
    Col 6. Body mass index (weight in kg/(height in m)^2)
    Col 7. Diabetes pedigree function
    Col 8. Age (years)
    Col 9. Class variable (0 or 1)

'''

from numpy import loadtxt   #to handle the text in the dataset
from keras.models import Sequential     #it will create an empty stack of layers where we are going to store our hidden layers.
from keras.layers import Dense  #what type of layer you need convolutional or dense ..etc
from keras.models import model_from_json    #to save the model into json or load the model back from the json file

dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ",")     #each word is splitted by ,
x = dataset[:,0:8]  #first 8 columns
y = dataset[:,8]    #last column (class)

model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))    #adding the layers in model, 12 is the number of nodes in first hidden layer
model.add(Dense(8, activation = 'relu'))    #no. of nodes in second layer = 8, activation Fn - relu
model.add(Dense(1, activation = 'sigmoid')) #no. of nodes in third layer = 1, activation Fn = sigmoid
                                            #You can add as much layers as you want
#model.summary()    #it will give you the structure of your model

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])    #to check any error, to maintain the accuracy and prevent the loss, to optimize the model
                                                                                            #loass Fn - It is the objective that model will try to minimize
                                                                                            #optimizer - string identifier of an existing optimizer
                                                                                            #List of metrics - For any classification problem you will want to set this to metrics = ['accuracy']
                                                                                            
model.fit(x,y, epochs = 150, batch_size = 10)    #training, it will give the accuracy & loss value for each and every batch
_, accuracy = model.evaluate(x,y)
print("Accuracy : %.2f" % (accuracy*100))

model_json = model.to_json()        #to save the model as json file so that we don't have to train it again and again
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

print("Model saved to disk")

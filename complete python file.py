import numpy as np
import pandas as pd

import tensorflow as tf
import seaborn as sns 

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import time
import tensorflow_model_optimization as tfmot
import tempfile

def plot_evaluation(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric],'')
    plt.title('Training and Validation '+metric.capitalize())
    plt.xlabel("No. of Epochs")
    plt.ylabel(metric)
    plt.legend(["training"+metric,'validation'+metric])
    plt.show()

def plot_confusion_matrix(true_labels, predictions, activities):
  # convert labels and predictions back to index numbers
  true_labels_index = np.argmax(true_labels, axis = 1)
  predictions_index = np.argmax(predictions, axis = 1)
  # produce consuion matrix using sklearn 
  CM = confusion_matrix(true_labels_index, predictions_index)
  recall = recall_score (true_labels_index, predictions_index, average = "macro")
  f1 = f1_score (true_labels_index, predictions_index, average = "macro")
  precision = precision_score(true_labels_index, predictions_index, average = "macro")
  print("Recall = ",recall)
  print("Precision =", precision)
  print("F1 Score =",f1)
  Multi_CM = multilabel_confusion_matrix(true_labels_index, predictions_index)  
  print(Multi_CM.shape)
  print(Multi_CM)
  #(6,2,2)
  '''
  TP = 0 #1,1
  FN = 0 #1,0
  FP = 0 #0,1
  TN = 0 #0,0
  for i in range (6):
    TP += Multi_CM[i][1][1]
    FN += Multi_CM[i][1][0]
    FP += Multi_CM[i][0][1]
    TN += Multi_CM[i][0][0]
  return TP, FN, FP, TN
  '''
  #produce sns heatmap -> fmt = 'd' Integer format ; pass list in to act as axis labels
  plt.figure(figsize=(20,16))
  sns.set(font_scale=2.2)
  sns.heatmap(CM, cmap='Blues' , annot = True, fmt = 'd', 
              xticklabels = activities, yticklabels = activities)
  # configure plt figure
  plt.xlabel("Predicted Class",fontsize=30)
  plt.ylabel("True Class", fontsize =30)
  plt.show()


# Using Spec Data
#trainX = np.load('train_Spec.npy')
#testX = np.load('test_Spec.npy')
trainX = np.load('train_Image.npy')
testX = np.load('test_Image.npy')
'''
trainX = np.load('trainX4.npy')
testX = np.load('testX4.npy')
'''
#print(trainX.shape)

trainX = trainX /255
testX = testX /255
# end is here



trainy = pd.read_csv('HARDataset/train/y_train.txt', sep='\s+', header=None).values
testy = pd.read_csv('HARDataset/test/y_test.txt', sep='\s+', header=None).values

trainy = trainy -1
testy = testy -1

trainX = trainX.astype('float32')
trainy = trainy.astype('float32')
testX = testX.astype('float32')
testy = testy.astype('float32')

# one-hot encoding
trainy = tf.keras.utils.to_categorical(trainy)
testy = tf.keras.utils.to_categorical(testy)

data_shape = trainX.shape[1:]
data_test_shape = testX.shape[1:]
print(data_shape)
print(data_test_shape)

n_series = 1
gru_nodes = 10
rate = 0.2
rate2 = 0.5
#build CNN network
# how does kernel shape, strides, & max_pool_size matter
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size

model.add(tf.keras.layers.Dropout(rate))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size
'''
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size


model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),  activation='relu', padding='same',input_shape=data_shape))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),  padding ='same')) # stride defaults to pool size
'''

# why do we use dropout
model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(rate2))
model.add(tf.keras.layers.Dense(6, activation='softmax'))
# dense is the number of labels we have

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

print(trainX.shape, trainy.shape)


#new
modelfit_1 = time.time()
history = model.fit(trainX,trainy, batch_size = 64, epochs = 1, 
                    validation_split = 0.2, shuffle = True, use_multiprocessing = True)
#testX=testX.reshape(2947,128,9,1)

modelfit_2 = time.time()
totalTime = modelfit_2 - modelfit_1
print("Total Time = ",totalTime)

#Evaluate model

test_loss, test_acc = model.evaluate(testX, testy, batch_size = 128)



model.summary()

_, model_accuracy = model.evaluate(
   testX, testy, verbose=0)
   
#Save model
name = input("Save model name: ")
modelpath='saved_models/'+name+'_model'
model.save(modelpath)
tf.keras.models.save_model(model, "model.h5")
# mmodel is saved as HDF5 format

plot_evaluation(history, 'accuracy')
plot_evaluation(history, 'loss')


#Plot Confusion Matrix on Test set
predictions = model.predict(testX)
Activities = ['Walking', 'Walking Up', 'Walking Down',
              'Sitting', 'Standing', 'Laying']


plot_confusion_matrix(testy,predictions, Activities)


print("TP = ", TP)
print("FN = ", FN)
print("FP = ", FP)
print("TN = ", TN)
Precision = TP/ (TP+FP)
Recall = TP/(TP+FN)
F1 = 2*(Precision*Recall)/(Precision+Recall)
print("Precision = ", Precision)
print("Recall = ",Recall)
print ("F1 Score = ", F1)

#name = "final_1_layer_image_model"
#Normal Model

converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/'+name+'_model')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()


#quantised dynamic range
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/'+name+'_model')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.experimental_enable_resource_variables = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()


#float 16 quantization
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/'+name+'_model')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.experimental_enable_resource_variables = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model_quant_float16 = converter.convert()


#Integer-Only Quantization Input Output 8 bits
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(trainX).batch(1).take(7000):
    # Model has only one input so each data point has one element.
    yield [input_value]
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/'+name+'_model')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.experimental_enable_resource_variables = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant_uint8 = converter.convert()

# Save the model.
#The wb indicates that the file is opened for writing in binary mode
# open a file called model.tflite to be written in binary mode
# wb write binary, model.tflite, file name

with open('test11.tflite', 'wb') as f:
    f.write(tflite_model)

with open('test22.tflite', 'wb') as f:
    f.write(tflite_quant_model)

with open('test33.tflite', 'wb') as f:
    f.write(tflite_model_quant_uint8)


with open('test44.tflite', 'wb') as f:
    f.write(tflite_model_quant_float16)


#load model into interpreter
#1 NO Optimisation
interpreter = tf.lite.Interpreter(model_path="test11.tflite")
interpreter.allocate_tensors()

#2 Dynamic Range Quantisation
interpreter_quant = tf.lite.Interpreter(model_path="test22.tflite")
interpreter_quant.allocate_tensors()

#3 Integer Quantisation
interpreter_int_quant = tf.lite.Interpreter(model_path="test33.tflite")
interpreter_int_quant.allocate_tensors()

#4 Float16 Quantisation
interpreter_float16_quant = tf.lite.Interpreter(model_path="test44.tflite")
interpreter_float16_quant.allocate_tensors()

#test model

# convert labels to index number from one-hot
testy = np.argmax(testy, axis = 1)

# 1 NO Optimisation 
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Run Interpreter in for loop for every test data entry 
accuracy_count = 0
#Start Time
time1 = time.time()
#test data
for i in range(testX.shape[0]): #number of samples of testing data
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], testX[i:i+1,:,:,:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predict_label = np.argmax(output_data) #predicted label from interpreter
    
    #check if prediction is correct
    if predict_label == testy[i]:
        accuracy_count += 1
        
#Overall accuracy for entire test 
accuracy = accuracy_count * 1.0 / testy.size 
print(accuracy_count)
print('No Optimistion accuracy = %.4f' % accuracy)
time2 = time.time()
print('No Optimistion Consuming time is %.4f' % (time2-time1))



#2 Dynamic Quantisation
# Get input and output tensors
input_details = interpreter_quant.get_input_details()
output_details = interpreter_quant.get_output_details()

#Run Interpreter in for loop for every test data entry 
accuracy_count = 0
# Start Time
time1 = time.time()
#test
for i in range(testX.shape[0]): #number of samples of testing data
    input_shape = input_details[0]['shape']
    interpreter_quant.set_tensor(input_details[0]['index'], testX[i:i+1,:,:,:])
    interpreter_quant.invoke()
    output_data = interpreter_quant.get_tensor(output_details[0]['index'])
    predict_label = np.argmax(output_data) #predicted label from interpreter
    
    #check if prediction is correct
    if predict_label == testy[i]:
        accuracy_count += 1
        
#Overall accuracy for entire test 
accuracy = accuracy_count * 1.0 / testy.size 
print(accuracy_count)
print('Dynamic Quantisation model accuracy = %.4f' % accuracy)
time2 = time.time()
print('Dynamic Quantisation Consuming time is %.4f' % (time2-time1))

#3 Float 16
# Get input and output tensors
input_details = interpreter_float16_quant.get_input_details()
output_details = interpreter_float16_quant.get_output_details()

#Run Interpreter in for loop for every test data entry 
accuracy_count = 0
# start time
time1 = time.time()
#test
for i in range(testX.shape[0]): #number of samples of testing data
    input_shape = input_details[0]['shape']
    interpreter_float16_quant.set_tensor(input_details[0]['index'], testX[i:i+1,:,:,:])
    interpreter_float16_quant.invoke()
    output_data = interpreter_float16_quant.get_tensor(output_details[0]['index'])
    predict_label = np.argmax(output_data) #predicted label from interpreter
    
    #check if prediction is correct
    if predict_label == testy[i]:
        accuracy_count += 1
        
#Overall accuracy for entire test 
accuracy = accuracy_count * 1.0 / testy.size 
print(accuracy_count)
print('Float16 Quantisation model accuracy = %.4f' % accuracy)
time2 = time.time()
print('Float16 Quantisation Consuming time is %.4f' % (time2-time1))

#4 Integer Quantisation

# Get input and output tensors

input_details = interpreter_int_quant.get_input_details()
output_details = interpreter_int_quant.get_output_details()

#Run Interpreter in for loop for every test data entry 
accuracy_count = 0
# Start Time
time1 = time.time()
# Check if the input type is quantized, then rescale input data to uint8
if input_details[0]['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details[0]["quantization"]
    testX = testX / input_scale + input_zero_point
#testX[i:i+1,:,:,:]  gives each each sample

# change testX -> Uint8
testX = testX.astype('uint8')
for i in range(testX.shape[0]): #number of samples of testing data
    input_shape = input_details[0]['shape']
    interpreter_int_quant.set_tensor(input_details[0]['index'], testX[i:i+1,:,:,:])
    interpreter_int_quant.invoke()
    output_data = interpreter_int_quant.get_tensor(output_details[0]['index'])
    # added [0] at end
    predict_label = np.argmax(output_data) #predicted label from interpreter
    
    #check if prediction is correct
    if predict_label == testy[i]:
        accuracy_count += 1
        
#Overall accuracy for entire test 
accuracy = accuracy_count * 1.0 / testy.size 
print(accuracy_count)
print('Integer Quantisation model accuracy = %.4f' % accuracy)
time2 = time.time()
print('Integer Quantisation Consuming time is %.4f' % (time2-time1))


# apply pruning on model
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 50
validation_split = 0.2 # 20% of training set will be used for validation set. 

num_images = trainX.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.50,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              # https://stackoverflow.com/questions/49161174/tensorflow-logits-and-labels-must-have-the-same-first-dimension
              # use sparse_categorical_crossentropy for 1D 
              # use categorical_crossentropy for 2D
              metrics=['accuracy'])

# log the differences
logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]


model_for_pruning.fit(trainX, trainy,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

# convert from (0,1) -> (0,6)
testX = np.load('test_Spec.npy')
testX = testX.astype('float32')
testy = tf.keras.utils.to_categorical(testy)

_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   testX, testy, verbose=0)



print('Baseline test accuracy:', model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)

model_for_pruning_stripped = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

converter_prune = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_stripped)
tflite_model_prune = converter_prune.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(tflite_model_prune)

print('Saved pruned TFLite model to:', pruned_tflite_file)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_pruning_stripped , pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)



def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

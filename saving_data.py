import numpy as np
import pandas as pd

import tensorflow as tf

from PIL import Image



def get_xn(Xs,n):
    # input is list
    '''
    calculate the Fourier coefficient X_n of 
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    #
    #print(xn)
    # Xs *e^(1j*2*pi*ks*n/L)
    return(xn)

def get_xns(ts):
    # input is list
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2, 
    to account for the symetry of the Fourier coefficients above the Nyquest Limit. 
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): # Nyquest Limit
        mag.append(np.abs(get_xn(ts,n))*2)
    #print(mag)
    return(mag)

def create_spectrogram(ts,NFFT,noverlap = None):
    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128. 
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    #print (starts)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < (len(ts)+1)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT]) 
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts) 
    return(starts,spec)

def convert_image(data, image,labels,i,version):
    # convert [64][9] -> [63][64][3] [H][W]
    u = 7
    for x in range (64):
        for y in range (9): # write in the values
            # read value and determine how high we go
            for z in range(u):
                # z gives values between 0-128
                # change these values = 1
                image[x][z+u*y][0] = image[x][z+u*y][1] = image[x][z+u*y][2] = data[x][0][y]*255
    #image = np.moveaxis(image, 1,0)

                # should be == 255 -> To make white, but we normalise value
    # save image into folder
    
    img = Image.fromarray(image,'RGB')
    if (version ==0 ):
        img.save('image_path_2/Label{}/pic{}.jpg'.format(labels[i][0],i))
    else:
        img.save('image_path_3/Label{}/pic{}.jpg'.format(labels[i][0],i))
    
    return image

the_list =[]
the_list2 =[]
for i in range(7352):
    the_list.append(np.zeros((64,63,3),dtype='uint8')) # create (7352,1152,128,3)

for i in range(2947):
    the_list2.append(np.zeros((64,63,3),dtype='uint8')) # create (7352,1152,128,3)

body_acc_x_train = pd.read_csv('HARDataset/train/Inertial Signals/body_acc_x_train.txt', sep='\s+', header=None)
body_acc_y_train = pd.read_csv('HARDataset/train/Inertial Signals/body_acc_y_train.txt', sep='\s+', header=None)
body_acc_z_train = pd.read_csv('HARDataset/train/Inertial Signals/body_acc_z_train.txt', sep='\s+', header=None)

total_acc_x_train = pd.read_csv('HARDataset/train/Inertial Signals/total_acc_x_train.txt', sep='\s+', header=None)
total_acc_y_train = pd.read_csv('HARDataset/train/Inertial Signals/total_acc_y_train.txt', sep='\s+', header=None)
total_acc_z_train = pd.read_csv('HARDataset/train/Inertial Signals/total_acc_z_train.txt', sep='\s+', header=None)

body_gyro_x_train = pd.read_csv('HARDataset/train/Inertial Signals/body_gyro_x_train.txt', sep='\s+', header=None)
body_gyro_y_train = pd.read_csv('HARDataset/train/Inertial Signals/body_gyro_y_train.txt', sep='\s+', header=None)
body_gyro_z_train = pd.read_csv('HARDataset/train/Inertial Signals/body_gyro_z_train.txt', sep='\s+', header=None)

# TEST
body_acc_x_test = pd.read_csv('HARDataset/test/Inertial Signals/body_acc_x_test.txt', sep='\s+', header=None)
body_acc_y_test = pd.read_csv('HARDataset/test/Inertial Signals/body_acc_y_test.txt', sep='\s+', header=None)
body_acc_z_test = pd.read_csv('HARDataset/test/Inertial Signals/body_acc_z_test.txt', sep='\s+', header=None)

total_acc_x_test = pd.read_csv('HARDataset/test/Inertial Signals/total_acc_x_test.txt', sep='\s+', header=None)
total_acc_y_test = pd.read_csv('HARDataset/test/Inertial Signals/total_acc_y_test.txt', sep='\s+', header=None)
total_acc_z_test = pd.read_csv('HARDataset/test/Inertial Signals/total_acc_z_test.txt', sep='\s+', header=None)

body_gyro_x_test = pd.read_csv('HARDataset/test/Inertial Signals/body_gyro_x_test.txt', sep='\s+', header=None)
body_gyro_y_test = pd.read_csv('HARDataset/test/Inertial Signals/body_gyro_y_test.txt', sep='\s+', header=None)
body_gyro_z_test = pd.read_csv('HARDataset/test/Inertial Signals/body_gyro_z_test.txt', sep='\s+', header=None)

trainy = pd.read_csv('HARDataset/train/y_train.txt', sep='\s+', header=None).values
testy = pd.read_csv('HARDataset/test/y_test.txt', sep='\s+', header=None).values

# turn into NP Array
body_acc_x_train= np.asarray(body_acc_x_train)
body_acc_y_train= np.asarray(body_acc_y_train)
body_acc_z_train= np.asarray(body_acc_z_train)
total_acc_x_train= np.asarray(total_acc_x_train)
total_acc_y_train= np.asarray(total_acc_y_train)
total_acc_z_train= np.asarray(total_acc_z_train)
body_gyro_x_train = np.asarray(body_gyro_x_train)
body_gyro_y_train = np.asarray(body_gyro_y_train)
body_gyro_z_train = np.asarray(body_gyro_z_train)

body_acc_x_test= np.asarray(body_acc_x_test)
body_acc_y_test= np.asarray(body_acc_y_test)
body_acc_z_test= np.asarray(body_acc_z_test)
total_acc_x_test= np.asarray(total_acc_x_test)
total_acc_y_test= np.asarray(total_acc_y_test)
total_acc_z_test= np.asarray(total_acc_z_test)
body_gyro_x_test = np.asarray(body_gyro_x_test)
body_gyro_y_test = np.asarray(body_gyro_y_test)
body_gyro_z_test = np.asarray(body_gyro_z_test)


data_body_acc_x_train= []
data_body_acc_y_train= []
data_body_acc_z_train= []
data_total_acc_x_train= []
data_total_acc_y_train= []
data_total_acc_z_train= []
data_body_gyro_x_train =[]
data_body_gyro_y_train =[]
data_body_gyro_z_train =[]

data_body_acc_x_test= []
data_body_acc_y_test= []
data_body_acc_z_test= []
data_total_acc_x_test= []
data_total_acc_y_test= []
data_total_acc_z_test= []
data_body_gyro_x_test =[]
data_body_gyro_y_test =[]
data_body_gyro_z_test =[]



#obtain spectrogram of each
for x in range (7352):
    d, bax = create_spectrogram(body_acc_x_train[x, :],128)
    data_body_acc_x_train.append(bax)
    d, bay = create_spectrogram(body_acc_y_train[x, :],128)
    data_body_acc_y_train.append(bay)
    d, baz = create_spectrogram(body_acc_z_train[x, :],128)
    data_body_acc_z_train.append(baz)
    d, tax = create_spectrogram(total_acc_x_train[x, :],128)
    data_total_acc_x_train.append(tax)
    d, tay = create_spectrogram(total_acc_y_train[x, :],128)
    data_total_acc_y_train.append(tay)
    d, taz = create_spectrogram(total_acc_z_train[x, :],128)
    data_total_acc_z_train.append(taz)
    d, gax = create_spectrogram(body_gyro_x_train[x, :],128)
    data_body_gyro_x_train.append(gax)
    d, gay = create_spectrogram(body_gyro_y_train[x, :],128)
    data_body_gyro_y_train.append(gay)
    d, gaz = create_spectrogram(body_gyro_z_train[x, :],128)
    data_body_gyro_z_train.append(gaz)
    print(x)


for x in range (2947):
    d, bax = create_spectrogram(body_acc_x_test[x, :],128)
    data_body_acc_x_test.append(bax)
    d, bay = create_spectrogram(body_acc_y_test[x, :],128)
    data_body_acc_y_test.append(bay)
    d, baz = create_spectrogram(body_acc_z_test[x, :],128)
    data_body_acc_z_test.append(baz)
    d, tax = create_spectrogram(total_acc_x_test[x, :],128)
    data_total_acc_x_test.append(tax)
    d, tay = create_spectrogram(total_acc_y_test[x, :],128)
    data_total_acc_y_test.append(tay)
    d, taz = create_spectrogram(total_acc_z_test[x, :],128)
    data_total_acc_z_test.append(taz)
    d, gax = create_spectrogram(body_gyro_x_test[x, :],128)
    data_body_gyro_x_test.append(gax)
    d, gay = create_spectrogram(body_gyro_y_test[x, :],128)
    data_body_gyro_y_test.append(gay)
    d, gaz = create_spectrogram(body_gyro_z_test[x, :],128)
    data_body_gyro_z_test.append(gaz)
    print(x)

# append all together
trainX = np.stack((data_body_acc_x_train, data_body_acc_y_train, data_body_acc_z_train,\
                   data_total_acc_x_train, data_total_acc_y_train, data_total_acc_z_train,\
                    data_body_gyro_x_train, data_body_gyro_y_train, data_body_gyro_z_train),axis=-1)

testX = np.stack((data_body_acc_x_test, data_body_acc_y_test, data_body_acc_z_test,\
                   data_total_acc_x_test, data_total_acc_y_test, data_total_acc_z_test,\
                    data_body_gyro_x_test, data_body_gyro_y_test, data_body_gyro_z_test),axis=-1)
    
trainy = trainy -1
testy = testy -1

# one-hot encoding
#trainy = tf.keras.utils.to_categorical(trainy)
#testy = tf.keras.utils.to_categorical(testy)

trainX = trainX.astype('float32')
trainy = trainy.astype('float32')
testX = testX.astype('float32')
testy = testy.astype('float32')



testX = tf.divide(
   tf.subtract(
      testX, 
      tf.reduce_min(testX)
   ), 
   tf.subtract(
      tf.reduce_max(testX), 
      tf.reduce_min(testX)
   )
)
trainX = tf.divide(
   tf.subtract(
      trainX, 
      tf.reduce_min(trainX)
   ), 
   tf.subtract(
      tf.reduce_max(trainX), 
      tf.reduce_min(trainX)
   )
)


print (tf.reduce_min(testX))
print (tf.reduce_max(testX))
print (tf.reduce_min(trainX))
print (tf.reduce_max(trainX))

trainX = trainX.numpy()

testX = testX.numpy()



for i in range (7352):
    the_list[i] = convert_image(trainX[i],the_list[i],trainy,i,0)
    print(i)



for i in range (2947):
    the_list2[i] = convert_image(testX[i],the_list2[i],testy,i,1)
    print(i)

    
trainX = np.stack(the_list)
testX = np.stack(the_list2)

trainX = trainX.tolist()
testX = testX.tolist()

trainX = np.asarray(trainX)
testX = np.asarray(testX)

#trainX = trainX.numpy()
#testX = testX.numpy()

np.save('train_Image', trainX)
np.save('test_Image', testX)

'''

np.save('train_Spec', trainX)
np.save('test_Spec', testX)
'''
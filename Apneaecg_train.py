import csv, os, random, sys, os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils,plot_model
import pandas as pd
#import wfdb
from models import cnnlstm
import json
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
# from torchsummary import summary
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
from sklearn import preprocessing
from keras import optimizers
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)

def Count_data_number(path):
    countA = 0
    countN = 0
    for dirPath, dirNames, fileNames in os.walk(path+"/A/"):

        # print(fileNames)
        for f in fileNames:
            countA += 1
    for dirPath, dirNames, fileNames in os.walk(path+"/N/"):

        # print(fileNames)
        for f in fileNames:
            countN += 1

    return int(countA),int(countN)
def DA_Jitter(X, sigma=0.05):
    Noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+Noise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    #print(X.shape[0],X.shape[1])
    #sys.exit()
    #test_x=(np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1))))
    #print("test_x_shape",np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()

    #print("xx",xx.shape)
    #sys.exit()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    #print("yy",yy)
    #sys.exit()
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    #print(cs_x)
    # cs_y = CubicSpline(xx[:,1], yy[:,1])
    # cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range)]).transpose()
def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)
def DA_TimeWarp(X, sigma=0.2):

    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])

    return X_new
def DistortTimesteps(X, sigma=0.2):

    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    #print(tt.shape)
    #sys.exit()
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    #print(tt_cum)

    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0]]
    #print(tt_cum[-1,0])
    #sys.exit()
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]

    return tt_cum
def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))
if __name__ == '__main__':

    train_data_path=os.getcwd()+"/data/physiobank_databasev3/"+"train/"
    trainA, trainN=Count_data_number(train_data_path)
    ###############################
    ##        process train data  ##
    ###############################
    print("process train data")

    traindataN1 = np.load('traindataN1.npy')
    traindataN_s = np.load('traindataN_s.npy')
    traindataA1 = np.load('traindataA1.npy')
    traindataA_s = np.load('traindataA_s.npy')
    #traindataA_test= np.load('traindataA2.npy')

    #print(traindataA1,"\ntest",traindataA_test)
    #sys.exit()
    traindataN2 = []
    traindataN2_s=[]
    for i in range(trainN):
        train_x = np.reshape(traindataN1[i], (-1, 6000))
        train_x_s=np.reshape(traindataN_s[i], (-1, 6000))
        traindataN2.append(train_x)
        traindataN2_s.append(train_x_s)
        #print(i)
    traindataN2 = np.array(traindataN2)
    traindataN2_s=np.array(traindataN2_s)
    traindataA2 = []
    traindataA2_s=[]
    for i in range(trainA):
        train_x = np.reshape(traindataA1[i], (-1, 6000))
        train_x_s=np.reshape(traindataA_s[i],(-1,6000))
        traindataA2.append(train_x)
        traindataA2_s.append(train_x_s)
    traindataA2 = np.array(traindataA2)
    traindataA2_s=np.array(traindataA2_s)

    ###############################
    ##     bulid data for train  ##
    ###############################
    label1= np.zeros(trainA) #label2 = np.ones(trainN-2123+1)
    label3 = np.ones(trainN)

    traindataA2_wrap5 = []
    for i in range(len(traindataA2)):
        traindataA2_tmp=traindataA2[i].transpose((1,0))
        traindataA2_wrap5.append(DA_TimeWarp(traindataA2_tmp, sigma=0.5))

    traindataA2_wrap5=np.array(traindataA2_wrap5)



    traindataA2_wrap5=traindataA2_wrap5.transpose((0,2,1))
    #print(traindataA2_wrap5.shape, traindataA2.shape)
    #sys.exit()
    y=np.concatenate((label3,label3,label1,label1),axis=0)
    print(y)
    X = np.concatenate((traindataN2,traindataN2_s,traindataA2, traindataA2_s), axis=0)
    #X= preprocessing.scale(X)
    X_train, X_vaild, y_train, y_vaild =train_test_split(X, y, test_size=0.1,random_state=52)

    X_train_wrap=[]
    X_train_wrap3 = []
    X_train_wrap4 = []
    X_train_wrap5 = []
    X_train_noise_warping=[]
    X_train_noise_warping2=[]

    X_train_noise=DA_Jitter(X_train,sigma = 0.2)
    X_train_noise2 = DA_Jitter(X_train, sigma=0.3)
    X_train_noise3 = DA_Jitter(X_train, sigma=0.4)
    #X_train_rot=DistortTimesteps(X_train)

    #MagWarp = DA_MagWarp(X_train, sigma=0.3)

    for i in range(len(X_train)):
        X_train_tmp=X_train[i].transpose((1,0))
        # X_train_noise_tmp=X_train_noise3[i].transpose((1,0))
        # X_train_noise_tmp2=X_train_noise2[i].transpose((1,0))
        #print(X_train_tmp.shape)
        #sys.exit()
        X_train_wrap.append(DA_TimeWarp(X_train_tmp,sigma = 0.2))
        X_train_wrap3.append(DA_TimeWarp(X_train_tmp, sigma=0.3))
        X_train_wrap4.append(DA_TimeWarp(X_train_tmp, sigma=0.4))
        X_train_wrap5.append(DA_TimeWarp(X_train_tmp, sigma=0.5))
        #X_train_noise_warping.append(DA_TimeWarp(X_train_noise_tmp,sigma=0.5))
        #X_train_noise_warping2.append(DA_TimeWarp(X_train_noise_tmp2,sigma=0.5))

    X_train_wrap=np.array(X_train_wrap)
    X_train_wrap3 = np.array(X_train_wrap3)
    X_train_wrap4= np.array(X_train_wrap4)
    X_train_wrap5 = np.array(X_train_wrap5)
    X_train_noise_warping=np.array(X_train_noise_warping)
    X_train_noise_warping2=np.array(X_train_noise_warping2)

    X_train_wrap=X_train_wrap.transpose(0,2,1)
    X_train_wrap3 = X_train_wrap3.transpose(0, 2, 1)
    X_train_wrap4 = X_train_wrap4.transpose(0, 2, 1)
    X_train_wrap5 = X_train_wrap5.transpose(0, 2, 1)
    #X_train_noise_warping=X_train_noise_warping.transpose(0,2,1)
    #X_train_noise_warping2 = X_train_noise_warping2.transpose(0, 2, 1)
    # print(X_train.shape,X_train_wrap.shape)
    #
    # plt.plot(X_train[0][0])
    # #plt.plot(X_train_rot[0][0])
    # plt.plot(X_train_wrap[0][0])
    # plt.show()
    # sys.exit()

    X_train=np.concatenate((X_train,X_train_wrap),axis=0)
    #X_train = np.concatenate((X_train, X_train_wrap,X_train_wrap3,X_train_wrap4,X_train_wrap5), axis=0)
    # #print(X_train.shape)
    #sys.exit()
    X_train=X_train.transpose((0,2,1))
    X_vaild=X_vaild.transpose((0,2,1))
    y_train=np.concatenate((y_train,y_train),axis=0)

    train_y_onehot = np_utils.to_categorical(y_train)
    valid_y_onehot = np_utils.to_categorical(y_vaild)
    input_shape=(6000,1)
    #optimizer = Adam(0.00002, 0.5)
   #####Train


    class_w = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)
    optimizers=optimizers.Nadam()#.sgd(momentum=0.9)#

    #sys.exit()
    ###############################
    ##     Training              ##
    ###############################

    for epoch in range(4):

        ###############################
        ##     bulid train model     ##
        ###############################
        dataset_name = "apnea_ecg"
        apnea_ecgname = dataset_name + '_cnn_v21'
        save_models_path = os.getcwd() + "/models_save/apnea20190801/"
        model_names = save_models_path + apnea_ecgname+'.'+str(epoch) + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
        callbacks_list = [model_checkpoint]

        ####### train model ########
        train_model = cnnlstm.cnn_v4(input_shape=input_shape, num_classes=2)

        train_model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])#categorical_crossentropy#binary_crossentropy


        train_history = train_model.fit(X_train, train_y_onehot, batch_size=(256), epochs=300,validation_data=(X_vaild,valid_y_onehot),shuffle=True
                                        ,class_weight=class_w,callbacks=callbacks_list)

        scores = train_model.evaluate(X_vaild, valid_y_onehot)
        print(scores)
        print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))
        predictions = train_model.predict(X_vaild)
        # self.cnn.predict(
        predictions = np.argmax(predictions, axis=1)

        #print(predictions[600:700])
        print("\t[Info] Display Confusion Matrix:")
        print("%s\n" % pd.crosstab(y_vaild, predictions, rownames=['label'], colnames=['predict']))

        # # summarize history for accuracy
        plt.figure(1)
        plt.clf()
        plt.plot(train_history.history['acc'])
        plt.plot(train_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'vaild'], loc='upper left')
        save_file_name=os.getcwd()+'/fig_and_other/apnea_detection/'+ apnea_ecgname+'.'+str(epoch)
        #plt.savefig(save_file_name+ '_acc.png')
        #plt.show()
        ######plot loss#######

        plt.figure(2)
        plt.clf()
        plt.plot(train_history.history['loss'])
        plt.plot(train_history.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'vaild'], loc='upper right')
        #plt.savefig(save_file_name + '_loss.png')
        #print(save_loss_name)

        # plt.savefig(save_file_name+ '_loss.jpg')
        # with open(save_file_name+'acc_file.json', 'w') as f:
        #     json.dump(train_history.history['acc'], f)
        # with open(save_file_name+'loss_file.json', 'w') as f:
        #     json.dump(train_history.history['loss'], f)
        # with open(save_file_name+'valloss_file.json', 'w') as f:
        #     json.dump(train_history.history['val_loss'], f)
        # with open(save_file_name+'valacc_file.json', 'w') as f:
        #     json.dump(train_history.history['val_acc'], f)


    print("Finish!!!")








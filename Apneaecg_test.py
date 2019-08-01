import csv, os, random, sys, os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils,plot_model
import pandas as pd

from models import cnnlstm
from keras import backend as K
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
# from torchsummary import summary
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
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
def Load_train(path):

    countA=0
    countN=0
    countC=0
    data=[]
    dataN = []
    for dirPathA, dirNamesA, fileNamesA in os.walk(path+"/A/"):
        for f in fileNamesA:
            countA +=1
    for dirPathN, dirNamesN, fileNamesN in os.walk(path+"/N/"):
        for f in fileNamesN:
            countN +=1
    for dirPathN, dirNamesN, fileNamesC in os.walk(path+"/C/"):
        for f in fileNamesC:
            countC +=1
    #print(fileNamesA,countN)
    #print(fileNamesN[1132])
    for k in range(countA):
        with open(path +"/A/"+ fileNamesA[k], newline='', encoding='utf-8') as csvfile:
            ECGA = []
            label=[]
            #matrix = []
            reader = csv.reader(csvfile)
            for row in reader:
                ECGA.append(float(row[0]))
                #matrix=np.array(ECGA)
                #label.append((row[1]))
            #matrix.append(ECGA)
            #matrix.append(label)
            data.append(ECGA)
    data = np.array(data)
    #print(data)

    for k in range(countN):
        #if k<countN/4:
            with open(path + "/N/" + fileNamesN[k], newline='', encoding='utf-8') as csvfile:
                label = []
                #matrix = []
                ECGN = []
                reader = csv.reader(csvfile)
                for row in reader:
                    ECGN.append(float(row[0]))
                    #matrix = np.array(ECGN)
                #label=np.array(label)

                dataN.append(ECGN)

    dataN = np.array(dataN)

    #dataN=dataN.reshape(countN,6000)
    #print(dataN.shape)
    return  data,dataN

if __name__ == '__main__':


    test_data_path=os.getcwd()+"/data/UCCDB/"#os.getcwd()+"/data/physiobank_databasev3/"+"test/"#
    save_models_path=os.getcwd()+"/models_save/2/"

    testA, testN = Count_data_number(test_data_path)

    # testdataA, testdataN=Load_train(test_data_path)
    #
    # testdataN1 = []
    # testdataA1 = []
    # for i in range(testN):
    #     testdataN1.append(np.array(testdataN[i]))
    # for i in range(testA):
    #     testdataA1.append(np.array(testdataA[i]))
    #
    # testdataN1=np.array(testdataN1)
    # testdataA1 = np.array(testdataA1)
    # #print(testdataN1[458].shape)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler2 = preprocessing.MinMaxScaler()
    # testdataN1 = min_max_scaler.fit_transform(testdataN1)
    # testdataA1 = min_max_scaler2.fit_transform(testdataA1)
    # testdataN1=np.load('traindataN1.npy')
    # testdataA1 = np.load('traindataA1.npy')


    #testdataN1=np.load('testdataN1.npy')
    #testdataA1 = np.load('testdataA1.npy')

    testdataN1 = np.load('ucddb_testdataN1.npy')
    testdataA1 = np.load('ucddb_testdataA1.npy')

    # print(testdataN1.shape,testdataA1.shape)
    # sys.exit()
    #testdataN1 = preprocessing.normalize(testdataN1, norm='l2')
    #testdataA1 = preprocessing.normalize(testdataA1, norm='l2')
    testdataN2 = []
    for i in range(testN):
        train_x = np.reshape(testdataN1[i], (-1, 6000))
        testdataN2.append(train_x)
        #print(i)
    testdataN2 = np.array(testdataN2)

    testdataA2 = []
    for i in range(testA):
        train_x = np.reshape(testdataA1[i], (-1, 6000))
        testdataA2.append(train_x)
    testdataA2 = np.array(testdataA2)

    label1 = np.zeros(testA)  # label2 = np.ones(trainN-2123+1)
    label3 = np.ones(testN)

    y = np.concatenate((label3, label1), axis=0)


    X = np.concatenate((testdataN2, testdataA2), axis=0)
    X_test = X.transpose((0, 2, 1))
    Y_test_onehot = np_utils.to_categorical(y)

    input_shape = (6000, 1)
    test_model = cnnlstm.cnn_v4(input_shape=input_shape, num_classes=2)#####
    test_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #apnea_ecg_cnn_n0.3.0.39 - 0.99
    dataset_name = "apnea_ecg"
    apnea_ecgname = dataset_name + '_cnn_v20'
    test_model.load_weights(os.getcwd()+'/models_save/StandardScaler_cnn_combinfv1/'+apnea_ecgname+'.2.74-0.99.hdf5')######
    #UCCDB A:2072 N:7869
    test_score = test_model.evaluate(X_test, Y_test_onehot)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100 * test_score[1])
    predictions = test_model.predict(X_test)
    # self.cnn.predict(

    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    np.save('reslut.npy', predictions)
    print("\t[Info] Display Confusion Matrix:")
    print("%s\n" % pd.crosstab(y, predictions, rownames=['label'], colnames=['predict']))

    # image_arr = np.reshape(X_test[12],(-1,6000,1))
    # layer_1 = K.function([test_model.layers[0].input], [test_model.get_layer("ELU1").output])
    # layer_2 = K.function([test_model.layers[0].input], [test_model.get_layer("conv5").output])
    # f1 = layer_1([image_arr])[0]
    # f2=layer_2([image_arr])[0]
    # re = np.transpose(f1, (0, 2, 1))
    # re2 = np.transpose(f2, (0, 2, 1))
    # print(re.shape)
    # plt.figure(1)
    # for i in range(8):
    #     #print(re[0][i])
    #     plt.subplot(8, 1, i + 1)
    #     x=np.linspace(0,len(re[0][i])-1,num=len(re[0][i]), endpoint=True)
    #     x1=np.linspace(0,len(re[0][i])-1,num=len(re2[0][i]), endpoint=True)
    #
    #     plt.plot(x,re[0][i], '--', x1, re2[0][i],'o')
        #plt.imshow(re[0][i])  # ,cmap='gray'
    #plt.show()
    # plt.figure(2)
    # for i in range(8):
    #     #print(re[0][i])
    #     plt.subplot(8, 1, i + 1)
    #     x=np.arange(0,len(re2[0][i]))
    #     plt.plot(x,re2[0][i])
    #     #plt.imshow(re[0][i])  # ,cmap='gray'
    # plt.show()












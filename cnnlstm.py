import numpy as np
import tensorflow as tf
from keras.layers import AveragePooling2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D,GlobalMaxPooling1D

from keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,UpSampling1D
from keras.layers.advanced_activations import LeakyReLU,ReLU,Softmax,ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, Convolution1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils,plot_model
from keras.utils import to_categorical
from keras.layers import GRU, LSTM, Bidirectional,Add,Concatenate
from keras import backend as K
from keras.layers.core import Lambda

#class Net():
# def __init__(self):
#         self.rows = 240
#         self.col = 3
#         self.channel = 3
#         optimizer = Adam(0.0002, 0.5)
#
#         self.inputshape = (self.rows, self.col)

        #self.cnn = self.cnn_v1()
        #self.cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #plot_model(self.cnn,to_file='model.png')
        # self.lstm=self.build_lstm()
        # self.lstm.compile(loss='categorical_crossentropy',optimizer=optimizer,  metrics = ['accuracy'])
        # self.cnn
        # print(self.cnn.output)

def bulid_cnn(self,inputLayer):

        #Input_layer=Input(shape=(self.inputshape ),name='inputlayer')

        #convl_models
        conv1=Conv1D(64, kernel_size=4, strides=4, padding='same', input_shape=self.inputshape)(inputLayer)
        BatchNormalization1=BatchNormalization(momentum=0.8)(conv1)
        relu1=ELU(alpha=1)(BatchNormalization1)
        #drop1=Dropout(0.5)(relu1)
        #convFine = MaxPooling1D(pool_size=8, strides=1, name='fMaxP1')(relu1)
        conv2=Conv1D(128, kernel_size=4, padding='same')(relu1)
        BatchNormalization2 = BatchNormalization(momentum=0.8)(conv2)
        relu2 = ELU(alpha=1)(BatchNormalization2)

        # maxpooling1=MaxPooling1D(pool_size=2)(BatchNormalization2)
        # cnndrop1 = Dropout(0.2)(maxpooling1)

        conv3=Conv1D(128, kernel_size=4,  padding='same')(relu2)
        BatchNormalization3 = BatchNormalization(momentum=0.8)(conv3)
        relu3 = ELU(alpha=1)(BatchNormalization3)

        conv4 = Conv1D(128, kernel_size=4, padding='same')(relu3)
        BatchNormalization4 = BatchNormalization(momentum=0.8)(conv4)
        relu4 = ELU(alpha=1)(BatchNormalization4)

        # conv5 = Conv1D(128, kernel_size=4, strides=2, padding='valid',kernel_initializer='RandomNormal')(relu4)
        # BatchNormalization5 = BatchNormalization(momentum=0.8)(conv5)
        # relu5 = LeakyReLU(alpha=0.2)(BatchNormalization5)
        maxpooling2 = MaxPooling1D(pool_size=4,strides=4,name='fMaxP2')(relu4)

        convFc=GlobalAveragePooling1D()(maxpooling2)
        fineShape=convFc.get_shape()



        # network to learn coarse features
        conv2_1 = Conv1D(64, kernel_size=3, strides=4, padding='same', input_shape=self.inputshape)(inputLayer)
        BatchNormalization2_1 = BatchNormalization(momentum=0.8)(conv2_1)
        relu2_1 = ReLU()(BatchNormalization2_1)
        #convFine = MaxPooling1D(pool_size=4, strides=1, name='cMaxP1')(relu2_1)

        conv2_2 = Conv1D(128, kernel_size=3, padding='same')(relu2_1)
        BatchNormalization2_2 = BatchNormalization(momentum=0.8)(conv2_2)
        relu2_2 = ReLU()(BatchNormalization2_2)
        #Max_pool1=MaxPooling1D(pool_size=2)(relu2_2)

        conv2_3= Conv1D(128, kernel_size=3, padding='same')(relu2_2)
        BatchNormalization2_3 = BatchNormalization(momentum=0.8)(conv2_3)
        relu1=LeakyReLU(alpha=0.2)(BatchNormalization2_3)

        conv2_4 = Conv1D(128, kernel_size=3, padding='same')(relu1)
        BatchNormalization2_4 = BatchNormalization(momentum=0.8)(conv2_4)
        relu1 = LeakyReLU(alpha=0.2)(BatchNormalization2_4)

        Maxpooling1=MaxPooling1D(pool_size=4, strides=4, name='cMaxP2')(relu1)
        #relu2_2 = ReLU()(BatchNormalization2_2)
        convFc2 = GlobalAveragePooling1D()(Maxpooling1)


        coarseShape = convFc2.get_shape()
        mergecnn=Concatenate( name='merge')([convFc,convFc2])
        BatchNormalizationc=BatchNormalization(momentum=0.8)(mergecnn)
        # convoutLayer = Dropout(rate=0.5, name='Drop1')(mergecnn)
        #
        # ReshapeLayer = Reshape((1,int(3840)))(convoutLayer)
        # Bilstmlayer1 = Bidirectional(LSTM(1024, activation='relu', dropout=0.5, name='bLstm1'))(ReshapeLayer)
        # ReshapeLayer2 = Reshape((1 ,int(Bilstmlayer1.get_shape()[1])))(Bilstmlayer1)
        # Bilstmlayer2 = Bidirectional(LSTM(1024, activation='relu', dropout=0.5, name='bLstm2'))(ReshapeLayer2)
        # BilstmoutLayer = Dense(3, activation='softmax', name='outLayer')(Bilstmlayer2)


        # concatenate coarse and fine cnns
        #fc_merge=Dense(256)(mergecnn)

        # reshape1=Reshape((256,1))(mergecnn)
        # bilstm=Bidirectional(LSTM(128, input_shape=(128, 1), return_sequences=True),merge_mode='concat')(reshape1)
        # lstm=LSTM(256, return_sequences=False)(bilstm)
        # #B=Dense(1024)(bilstm)
        #
        # mergecnnlstm=Concatenate()([fc_merge,lstm])
        # dropout2=Dropout(0.5)(mergecnnlstm)
        # fc1=Dense(256,kernel_initializer='RandomNormal')(dropout2)
        # activityfc1=Activation('relu')(fc1)
        # dropoutfc1=Dropout(0.25)(activityfc1)
        #
        # fc2 = Dense(128,kernel_initializer='RandomNormal')(dropoutfc1)
        # activityfc2 = Activation('relu')(fc2)
        # dropoutfc2 = Dropout(0.25)(activityfc2)
        #
        #
        # fc3 = Dense(32,kernel_initializer='RandomNormal')(dropoutfc2)
        # activityfc3 = Activation('relu')(fc3)
        # dropoutfc3 = Dropout(0.25)(activityfc3)
        #
        # fc4 = Dense(3,kernel_initializer='RandomNormal')(dropoutfc3)
        # activityfc4 = Activation('softmax')(fc4)


        #Merged=Model(outputs=[BilstmoutLayer],inputs=[Input_layer])
        #Merged.summary()




        #return Merged
        return BatchNormalizationc, (coarseShape, fineShape)

def cnn_v1(input_shape, num_classes):

        fs=100
        Input_layer = Input(shape=(input_shape), name='inputlayer')

        #convl_models
        conv1 = Conv1D(64, kernel_size=int(fs/2), strides=int(fs/16),padding='same', input_shape=input_shape,
                       kernel_initializer='RandomNormal')(Input_layer)

        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = ELU(alpha=1)(conv1)
        maxpooling1=MaxPooling1D(pool_size=8,strides=8, name='fMaxP1')(conv1)

        drop1=Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2', kernel_initializer='RandomNormal')(drop1)
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = ELU(alpha=1)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3', kernel_initializer='RandomNormal')(conv2)
        conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3 = ELU(alpha=1)(conv3)
        conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4', kernel_initializer='RandomNormal')(conv3)
        conv4 = BatchNormalization(momentum=0.8)(conv4)
        conv4 = ELU(alpha=1)(conv4)

        conv5 =  Conv1D(filters=512, kernel_size=8, padding='same', name='fConv5', kernel_initializer='RandomNormal')(conv4)
        conv5 = BatchNormalization(momentum=0.8)(conv5)
        conv5 = ELU(alpha=1)(conv5)
        maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        globolPooling1 = GlobalMaxPooling1D()(maxpooling2)

        # define_conv2_models

        # conv2_1 = Conv1D(64, kernel_size=fs*4, strides=int(fs/2), padding='same', input_shape=input_shape,
        #                  kernel_initializer='RandomNormal', name='cConv1')(Input_layer)
        # conv2_1 = BatchNormalization(momentum=0.8)(conv2_1)
        # conv2_1 = ReLU()(conv2_1)
        # maxpoolingC=MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        # dropC=Dropout(rate=0.5, name='cDrop1')(maxpoolingC)
        #
        # conv2_2 = Conv1D(256, kernel_size=6, padding='same', name='cConv2', kernel_initializer='RandomNormal')(dropC)
        # conv2_2 = BatchNormalization(momentum=0.8)(conv2_2)
        # conv2_2 = ReLU()(conv2_2)
        # #Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)
        #
        # conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3', kernel_initializer='RandomNormal')(conv2_2)
        # conv2_3 = BatchNormalization(momentum=0.8)(conv2_3)
        # conv2_3 = ReLU()(conv2_3)
        # conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4', kernel_initializer='RandomNormal')(conv2_3)
        # conv2_4 = BatchNormalization(momentum=0.8)(conv2_4)
        # conv2_4 = ReLU()(conv2_4)
        #
        # conv2_5 = Conv1D(512, kernel_size=6, padding='same', activation='relu', name='cConv5', kernel_initializer='RandomNormal')(conv2_4)
        # conv2_5 = BatchNormalization(momentum=0.8)(conv2_5)
        # conv2_5 = ReLU()(conv2_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # #relu2_2 = ReLU()(BatchNormalization2_2)
        # globolPooling2 = GlobalMaxPooling1D()(maxpoolingC2)
        #
        # mergecnn = Concatenate()([globolPooling1, globolPooling2])

        # dropout3=Dropout(mergecnn)
        #fc_merge = Dense(512)(mergecnn)

        reshape1 = Reshape((1, 512))(globolPooling1)
        lstm = (LSTM(512,recurrent_dropout=0.5,input_shape=(512,1), return_sequences=True))(reshape1)

        lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=True)(lstm)
        lstm=Dropout(0.5)(lstm)
        lstm2 = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)

        fc1 = Dense(512, kernel_initializer='RandomNormal')(lstm2)
        activityfc1 = Activation('elu')(fc1)
        dropoutfc1 = Dropout(0.25)(activityfc1)

        fc2 = Dense(128, kernel_initializer='RandomNormal')(dropoutfc1)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.25)(activityfc2)

        fc3 = Dense(32, kernel_initializer='RandomNormal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.25)(activityfc3)

        fc4 = Dense(10, kernel_initializer='RandomNormal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='RandomNormal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        #Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged

def cnn_v2(input_shape, num_classes):

        fs=100
        Input_layer = Input(shape=(input_shape), name='inputlayer')

        #convl_models
        conv1 = Conv1D(64, kernel_size=int(fs/2), strides=int(fs/16),padding='same', input_shape=input_shape,
                       kernel_initializer='RandomNormal')(Input_layer)

        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = ELU(alpha=1)(conv1)
        maxpooling1=MaxPooling1D(pool_size=8,strides=8, name='fMaxP1')(conv1)

        drop1=Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2', kernel_initializer='RandomNormal')(drop1)
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = ELU(alpha=1)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3', kernel_initializer='RandomNormal')(conv2)
        conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3 = ELU(alpha=1)(conv3)
        conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4', kernel_initializer='RandomNormal')(conv3)
        conv4 = BatchNormalization(momentum=0.8)(conv4)
        conv4 = ELU(alpha=1)(conv4)

        conv5 =  Conv1D(filters=512, kernel_size=8, padding='same', name='fConv5', kernel_initializer='RandomNormal')(conv4)
        conv5 = BatchNormalization(momentum=0.8)(conv5)
        conv5 = ELU(alpha=1)(conv5)
        #maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        globolPooling1 = GlobalAveragePooling1D()(conv5)

        # define_conv2_models

        # conv2_1 = Conv1D(64, kernel_size=fs*4, strides=int(fs/2), padding='same', input_shape=input_shape,
        #                  kernel_initializer='RandomNormal', name='cConv1')(Input_layer)
        # conv2_1 = BatchNormalization(momentum=0.8)(conv2_1)
        # conv2_1 = ReLU()(conv2_1)
        # maxpoolingC=MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        # dropC=Dropout(rate=0.5, name='cDrop1')(maxpoolingC)
        #
        # conv2_2 = Conv1D(256, kernel_size=6, padding='same', name='cConv2', kernel_initializer='RandomNormal')(dropC)
        # conv2_2 = BatchNormalization(momentum=0.8)(conv2_2)
        # conv2_2 = ReLU()(conv2_2)
        # #Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)
        #
        # conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3', kernel_initializer='RandomNormal')(conv2_2)
        # conv2_3 = BatchNormalization(momentum=0.8)(conv2_3)
        # conv2_3 = ReLU()(conv2_3)
        # conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4', kernel_initializer='RandomNormal')(conv2_3)
        # conv2_4 = BatchNormalization(momentum=0.8)(conv2_4)
        # conv2_4 = ReLU()(conv2_4)
        #
        # conv2_5 = Conv1D(512, kernel_size=6, padding='same', activation='relu', name='cConv5', kernel_initializer='RandomNormal')(conv2_4)
        # conv2_5 = BatchNormalization(momentum=0.8)(conv2_5)
        # conv2_5 = ReLU()(conv2_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # #relu2_2 = ReLU()(BatchNormalization2_2)
        # globolPooling2 = GlobalMaxPooling1D()(maxpoolingC2)
        #
        # mergecnn = Concatenate()([globolPooling1, globolPooling2])

        # dropout3=Dropout(mergecnn)
        #fc_merge = Dense(512)(mergecnn)

        reshape1 = Reshape((1, 512))(globolPooling1)
        lstm = Bidirectional(LSTM(512,recurrent_dropout=0.5,input_shape=(512,1), return_sequences=True))(reshape1)

        lstm = (LSTM(512, recurrent_dropout=0.5,return_sequences=False))(lstm)

        #lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)

        fc1 = Dense(512, kernel_initializer='RandomNormal')(lstm)
        activityfc1 = Activation('elu')(fc1)
        dropoutfc1 = Dropout(0.25)(activityfc1)

        fc2 = Dense(128, kernel_initializer='RandomNormal')(dropoutfc1)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.25)(activityfc2)

        fc3 = Dense(32, kernel_initializer='RandomNormal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.25)(activityfc3)

        fc4 = Dense(10, kernel_initializer='RandomNormal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='RandomNormal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        #Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged
def cnn_v3(input_shape, num_classes):

        fs=100
        Input_layer = Input(shape=(input_shape), name='inputlayer')

        #convl_models
        conv1 = Conv1D(64, kernel_size=int(fs/2), strides=int(fs/16),padding='same', input_shape=input_shape)(Input_layer)

        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = ELU(alpha=1)(conv1)
        maxpooling1=MaxPooling1D(pool_size=8,strides=8, name='fMaxP1')(conv1)

        drop1=Dropout(0.5)(maxpooling1)
        #

        # conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2')(drop1)
        # conv2 = BatchNormalization(momentum=0.8)(conv2)
        # conv2 = ELU(alpha=1)(conv2)
        # conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3')(conv2)
        # conv3 = BatchNormalization(momentum=0.8)(conv3)
        # conv3 = ELU(alpha=1)(conv3)
        # conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4')(conv3)
        # conv4 = BatchNormalization(momentum=0.8)(conv4)
        # conv4 = ELU(alpha=1)(conv4)
        #
        # conv5 =  Conv1D(filters=512, kernel_size=8, padding='same', name='fConv5')(conv4)
        # conv5 = BatchNormalization(momentum=0.8)(conv5)
        # conv5 = ELU(alpha=1)(conv5)
        #maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        lstm1 = (LSTM(64,recurrent_dropout=0.5, input_shape=(1000,64),return_sequences=True))(conv1)

        lstm1 = LSTM(64, recurrent_dropout=0.5,return_sequences=False)(lstm1)

        #globolPooling1 = GlobalMaxPooling1D()(conv5)

        # define_conv2_models

        conv2_1 = Conv1D(64, kernel_size=fs*4, strides=int(fs/2), padding='same', input_shape=input_shape, name='cConv1')(Input_layer)
        conv2_1 = BatchNormalization(momentum=0.8)(conv2_1)
        conv2_1 = ReLU()(conv2_1)
        maxpoolingC=MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        dropC=Dropout(rate=0.5, name='cDrop1')(maxpoolingC)
        #
        # conv2_2 = Conv1D(128, kernel_size=6, padding='same', name='cConv2')(dropC)
        # conv2_2 = BatchNormalization(momentum=0.8)(conv2_2)
        # conv2_2 = ReLU()(conv2_2)
        # #Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)
        #
        # conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3')(conv2_2)
        # conv2_3 = BatchNormalization(momentum=0.8)(conv2_3)
        # conv2_3 = ReLU()(conv2_3)
        # conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4')(conv2_3)
        # conv2_4 = BatchNormalization(momentum=0.8)(conv2_4)
        # conv2_4 = ReLU()(conv2_4)
        #
        # conv2_5 = Conv1D(512, kernel_size=6, padding='same', activation='relu', name='cConv5')(conv2_4)
        # conv2_5 = BatchNormalization(momentum=0.8)(conv2_5)
        # conv2_5 = ReLU()(conv2_5)

        lstm2 = (LSTM(64, recurrent_dropout=0.5,input_shape=(120,64), return_sequences=True))(conv2_1)
        lstm2 = (LSTM(64, recurrent_dropout=0.5, return_sequences=False))(lstm2)
        #maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        #relu2_2 = ReLU()(BatchNormalization2_2)
        #globolPooling2 = GlobalMaxPooling1D()(conv2_5)

        mergecnn = Concatenate()([lstm1, lstm2])

        # dropout3=Dropout(mergecnn)
        #fc_merge = Dense(512)(mergecnn)

        #reshape1 = Reshape((1, 512))(globolPooling1)
        # lstm = (LSTM(512,recurrent_dropout=0.5, return_sequences=True))(maxpooling2)
        #
        # lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=True)(lstm)
        #
        # lstm2 = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)


        # fc1 = Dense(512, kernel_initializer='he_normal')(mergecnn)
        # activityfc1 = Activation('elu')(fc1)
        # dropoutfc1 = Dropout(0.25)(activityfc1)

        fc2 = Dense(128, kernel_initializer='he_normal')(mergecnn)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.25)(activityfc2)

        fc3 = Dense(32, kernel_initializer='he_normal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.25)(activityfc3)

        fc4 = Dense(10, kernel_initializer='he_normal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='he_normal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        #Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged
def cnn_v4(input_shape, num_classes):
        fs = 100
        Input_layer = Input(shape=(input_shape), name='inputlayer')
        # convl_models
        conv1 = Conv1D(64, kernel_size=int(fs / 2), strides=int(fs / 16), padding='same', input_shape=input_shape,name='fConv1')(
                Input_layer)

        conv1 = BatchNormalization(momentum=0.99,name='BatchNormal1')(conv1)
        conv1 = ELU(alpha=1,name='ELU1')(conv1)
        maxpooling1 = MaxPooling1D(pool_size=8, strides=8, name='fMaxP1')(conv1)

        drop1 = Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2')(drop1)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        conv2 = ELU(alpha=1)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3')(conv2)
        conv3 = BatchNormalization(momentum=0.99)(conv3)
        conv3 = ELU(alpha=1)(conv3)
        conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4')(conv3)
        conv4 = BatchNormalization(momentum=0.99)(conv4)
        conv4 = ELU(alpha=1)(conv4)

        conv5 = Conv1D(filters=512, kernel_size=8, padding='same', name='fConv5')(conv4)
        conv5 = BatchNormalization(momentum=0.99)(conv5)
        conv5 = ELU(alpha=1)(conv5)
        # maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        globolPooling1 = GlobalMaxPooling1D()(conv5)

        # define_conv2_models

        conv2_1 = Conv1D(64, kernel_size=fs * 4, strides=int(fs / 2), padding='same', input_shape=input_shape,
                         name='cConv1')(Input_layer)
        conv2_1 = BatchNormalization(momentum=0.99)(conv2_1)
        conv2_1 = ReLU()(conv2_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        dropC = Dropout(rate=0.5, name='cDrop1')(maxpoolingC)

        conv2_2 = Conv1D(128, kernel_size=6, padding='same', name='cConv2')(dropC)
        conv2_2 = BatchNormalization(momentum=0.99)(conv2_2)
        conv2_2 = ReLU()(conv2_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3')(conv2_2)
        conv2_3 = BatchNormalization(momentum=0.99)(conv2_3)
        conv2_3 = ReLU()(conv2_3)
        conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4')(conv2_3)
        conv2_4 = BatchNormalization(momentum=0.99)(conv2_4)
        conv2_4 = ReLU()(conv2_4)

        conv2_5 = Conv1D(512, kernel_size=6, padding='same', activation='relu', name='cConv5')(conv2_4)
        conv2_5 = BatchNormalization(momentum=0.99)(conv2_5)
        conv2_5 = ReLU()(conv2_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling2 = GlobalMaxPooling1D()(conv2_5)

        conv3_1 = Conv1D(64, kernel_size=fs , strides=int(fs/2), padding='same', input_shape=input_shape,
                         name='TConv1')(Input_layer)
        conv3_1 = BatchNormalization(momentum=0.99)(conv3_1)
        conv3_1 = LeakyReLU()(conv3_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='TMaxP1')(conv3_1)
        dropT = Dropout(rate=0.5, name='TDrop1')(maxpoolingC)

        conv3_2 = Conv1D(128, kernel_size=10, padding='same', name='TConv2')(dropT)
        conv3_2 = BatchNormalization(momentum=0.99)(conv3_2)
        conv3_2 = LeakyReLU()(conv3_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv3_3 = Conv1D(256, kernel_size=10, padding='same', name='TConv3')(conv3_2)
        conv3_3 = BatchNormalization(momentum=0.99)(conv3_3)
        conv3_3 = LeakyReLU()(conv3_3)
        conv3_4 = Conv1D(256, kernel_size=10, padding='same', name='TConv4')(conv3_3)
        conv3_4 = BatchNormalization(momentum=0.99)(conv3_4)
        conv3_4 = LeakyReLU()(conv3_4)

        conv3_5 = Conv1D(512, kernel_size=10, padding='same', activation='relu', name='TConv5')(conv3_4)
        conv3_5 = BatchNormalization(momentum=0.99)(conv3_5)
        conv3_5 = LeakyReLU()(conv3_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling3 = GlobalMaxPooling1D()(conv3_5)


        mergecnn = Concatenate()([globolPooling1, globolPooling2,globolPooling3])

        # dropout3=Dropout(mergecnn)
        # fc_merge = Dense(512)(mergecnn)

        # reshape1 = Reshape((1, 512))(globolPooling1)
        # lstm = (LSTM(512,recurrent_dropout=0.5, return_sequences=True))(maxpooling2)
        #
        # lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=True)(lstm)
        #
        # lstm2 = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)

        fc1 = Dense(512, kernel_initializer='he_normal')(mergecnn)
        activityfc1 = Activation('elu')(fc1)
        dropoutfc1 = Dropout(0.25)(activityfc1)

        fc2 = Dense(128, kernel_initializer='he_normal')(dropoutfc1)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.25)(activityfc2)

        fc3 = Dense(32, kernel_initializer='he_normal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.25)(activityfc3)

        fc4 = Dense(10, kernel_initializer='he_normal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='he_normal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        # Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged
def cnn_v5(input_shape, num_classes):
        fs = 100
        Input_layer = Input(shape=(input_shape), name='inputlayer')
        # convl_models
        conv1 = Conv1D(64, kernel_size=int(fs / 2), strides=int(fs / 16), padding='same', input_shape=input_shape,name='fConv1')(
                Input_layer)

        conv1 = BatchNormalization(momentum=0.99,name='BatchNormal1')(conv1)
        conv1 = ReLU(name='RELU1')(conv1)
        maxpooling1 = MaxPooling1D(pool_size=8, strides=8, name='fMaxP1')(conv1)

        drop1 = Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2')(drop1)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        conv2 = ReLU()(conv2)
        conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3')(conv2)
        conv3 = BatchNormalization(momentum=0.99)(conv3)
        conv3 = ReLU()(conv3)
        conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4')(conv3)
        conv4 = BatchNormalization(momentum=0.99)(conv4)
        conv4 = ReLU()(conv4)

        conv5 = Conv1D(filters=512, kernel_size=8, padding='same', name='fConv5')(conv4)
        conv5 = BatchNormalization(momentum=0.99)(conv5)
        conv5 = ReLU(name='conv5')(conv5)
        # maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        globolPooling1 = GlobalMaxPooling1D(name='globolPooling1')(conv5)

        # define_conv2_models

        conv2_1 = Conv1D(64, kernel_size=fs * 4, strides=int(fs / 2), padding='same', input_shape=input_shape,
                         name='cConv1')(Input_layer)
        conv2_1 = BatchNormalization(momentum=0.99)(conv2_1)
        conv2_1 = ReLU()(conv2_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        dropC = Dropout(rate=0.5, name='cDrop1')(maxpoolingC)

        conv2_2 = Conv1D(128, kernel_size=6, padding='same', name='cConv2')(dropC)
        conv2_2 = BatchNormalization(momentum=0.99)(conv2_2)
        conv2_2 = ReLU()(conv2_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3')(conv2_2)
        conv2_3 = BatchNormalization(momentum=0.99)(conv2_3)
        conv2_3 = ReLU()(conv2_3)
        conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4')(conv2_3)
        conv2_4 = BatchNormalization(momentum=0.99)(conv2_4)
        conv2_4 = ReLU()(conv2_4)

        conv2_5 = Conv1D(512, kernel_size=6, padding='same', activation='relu', name='cConv5')(conv2_4)
        conv2_5 = BatchNormalization(momentum=0.99)(conv2_5)
        conv2_5 = ReLU()(conv2_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling2 = GlobalMaxPooling1D()(conv2_5)

        conv3_1 = Conv1D(64, kernel_size=fs , strides=int(fs/2), padding='same', input_shape=input_shape,
                         name='TConv1')(Input_layer)
        conv3_1 = BatchNormalization(momentum=0.99)(conv3_1)
        conv3_1 = LeakyReLU()(conv3_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='TMaxP1')(conv3_1)
        dropT = Dropout(rate=0.5, name='TDrop1')(maxpoolingC)

        conv3_2 = Conv1D(128, kernel_size=10, padding='same', name='TConv2')(dropT)
        conv3_2 = BatchNormalization(momentum=0.99)(conv3_2)
        conv3_2 = LeakyReLU()(conv3_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv3_3 = Conv1D(256, kernel_size=10, padding='same', name='TConv3')(conv3_2)
        conv3_3 = BatchNormalization(momentum=0.99)(conv3_3)
        conv3_3 = LeakyReLU()(conv3_3)
        conv3_4 = Conv1D(256, kernel_size=10, padding='same', name='TConv4')(conv3_3)
        conv3_4 = BatchNormalization(momentum=0.99)(conv3_4)
        conv3_4 = LeakyReLU()(conv3_4)

        conv3_5 = Conv1D(512, kernel_size=10, padding='same', activation='relu', name='TConv5')(conv3_4)
        conv3_5 = BatchNormalization(momentum=0.99)(conv3_5)
        conv3_5 = LeakyReLU()(conv3_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling3 = GlobalMaxPooling1D()(conv3_5)


        mergecnn = Concatenate()([globolPooling1, globolPooling2,globolPooling3])

        # dropout3=Dropout(mergecnn)
        # fc_merge = Dense(512)(mergecnn)

        # reshape1 = Reshape((1, 512))(globolPooling1)
        # lstm = (LSTM(512,recurrent_dropout=0.5, return_sequences=True))(maxpooling2)
        #
        # lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=True)(lstm)
        #
        # lstm2 = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)
        # fc0 = Dense(1024, kernel_initializer='he_normal')(mergecnn)
        # activityfc0 = Activation('elu')(fc0)
        dropoutfc1 = Dropout(0.5)(mergecnn)

        fc1 = Dense(512, kernel_initializer='he_normal')(dropoutfc1)
        activityfc1 = Activation('elu')(fc1)
        dropoutfc1 = Dropout(0.5)(activityfc1)

        fc2 = Dense(128, kernel_initializer='he_normal')(dropoutfc1)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.5)(activityfc2)

        fc3 = Dense(64, kernel_initializer='he_normal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc3 = Dense(32, kernel_initializer='he_normal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(10, kernel_initializer='he_normal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='he_normal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        # Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged
def cnn_v6(input_shape, num_classes):
        fs = 100
        Input_layer = Input(shape=(input_shape), name='inputlayer')
        # convl_models
        conv1 = Conv1D(64, kernel_size=int(fs / 2), strides=int(fs / 16), padding='same', input_shape=input_shape,name='fConv1')(
                Input_layer)

        conv1 = BatchNormalization(momentum=0.99,name='BatchNormal1')(conv1)
        conv1 = ELU(alpha=1,name='ELU1')(conv1)
        maxpooling1 = MaxPooling1D(pool_size=8, strides=8, name='fMaxP1')(conv1)

        drop1 = Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=128, kernel_size=8, padding='same', name='fConv2')(drop1)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        conv2 = ELU(alpha=1)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv3')(conv2)
        conv3 = BatchNormalization(momentum=0.99)(conv3)
        conv3 = ELU(alpha=1)(conv3)
        conv4 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv4')(conv3)
        conv4 = BatchNormalization(momentum=0.99)(conv4)
        conv4 = ELU(alpha=1)(conv4)

        conv5 = Conv1D(filters=256, kernel_size=8, padding='same', name='fConv5')(conv4)
        conv5 = BatchNormalization(momentum=0.99)(conv5)
        conv5 = ELU(alpha=1)(conv5)
        # maxpooling2 = MaxPooling1D(pool_size=4, strides=4, name='fMaxP2')(conv5)

        globolPooling1 = GlobalMaxPooling1D()(conv5)

        # define_conv2_models

        conv2_1 = Conv1D(64, kernel_size=fs * 4, strides=int(fs / 2), padding='same', input_shape=input_shape,
                         name='cConv1')(Input_layer)
        conv2_1 = BatchNormalization(momentum=0.99)(conv2_1)
        conv2_1 = ReLU()(conv2_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='cMaxP1')(conv2_1)
        dropC = Dropout(rate=0.5, name='cDrop1')(maxpoolingC)

        conv2_2 = Conv1D(128, kernel_size=6, padding='same', name='cConv2')(dropC)
        conv2_2 = BatchNormalization(momentum=0.99)(conv2_2)
        conv2_2 = ReLU()(conv2_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv2_3 = Conv1D(256, kernel_size=6, padding='same', name='cConv3')(conv2_2)
        conv2_3 = BatchNormalization(momentum=0.99)(conv2_3)
        conv2_3 = ReLU()(conv2_3)
        conv2_4 = Conv1D(256, kernel_size=6, padding='same', name='cConv4')(conv2_3)
        conv2_4 = BatchNormalization(momentum=0.99)(conv2_4)
        conv2_4 = ReLU()(conv2_4)

        conv2_5 = Conv1D(256, kernel_size=6, padding='same', activation='relu', name='cConv5')(conv2_4)
        conv2_5 = BatchNormalization(momentum=0.99)(conv2_5)
        conv2_5 = ReLU()(conv2_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling2 = GlobalMaxPooling1D()(conv2_5)

        conv3_1 = Conv1D(64, kernel_size=fs , strides=int(fs/2), padding='same', input_shape=input_shape,
                         name='TConv1')(Input_layer)
        conv3_1 = BatchNormalization(momentum=0.99)(conv3_1)
        conv3_1 = LeakyReLU()(conv3_1)
        maxpoolingC = MaxPooling1D(pool_size=4, strides=4, name='TMaxP1')(conv3_1)
        dropT = Dropout(rate=0.5, name='TDrop1')(maxpoolingC)

        conv3_2 = Conv1D(128, kernel_size=10, padding='same', name='TConv2')(dropT)
        conv3_2 = BatchNormalization(momentum=0.99)(conv3_2)
        conv3_2 = LeakyReLU()(conv3_2)
        # Max_pool1 = MaxPooling1D(pool_size=2)(relu2_2)

        conv3_3 = Conv1D(256, kernel_size=10, padding='same', name='TConv3')(conv3_2)
        conv3_3 = BatchNormalization(momentum=0.99)(conv3_3)
        conv3_3 = LeakyReLU()(conv3_3)
        conv3_4 = Conv1D(256, kernel_size=10, padding='same', name='TConv4')(conv3_3)
        conv3_4 = BatchNormalization(momentum=0.99)(conv3_4)
        conv3_4 = LeakyReLU()(conv3_4)

        conv3_5 = Conv1D(256, kernel_size=10, padding='same', activation='relu', name='TConv5')(conv3_4)
        conv3_5 = BatchNormalization(momentum=0.99)(conv3_5)
        conv3_5 = LeakyReLU()(conv3_5)
        # maxpoolingC2=MaxPooling1D(pool_size=2, strides=2, name='cMaxP2')(conv2_5)
        # relu2_2 = ReLU()(BatchNormalization2_2)
        globolPooling3 = GlobalMaxPooling1D()(conv3_5)


        mergecnn = Concatenate()([globolPooling1, globolPooling2,globolPooling3])

        # dropout3=Dropout(mergecnn)
        # fc_merge = Dense(512)(mergecnn)

        # reshape1 = Reshape((1, 512))(globolPooling1)
        # lstm = (LSTM(512,recurrent_dropout=0.5, return_sequences=True))(maxpooling2)
        #
        # lstm = LSTM(512, recurrent_dropout=0.5,return_sequences=True)(lstm)
        #
        # lstm2 = LSTM(512, recurrent_dropout=0.5,return_sequences=False)(lstm)

        fc1 = Dense(512, kernel_initializer='he_normal')(mergecnn)
        activityfc1 = Activation('elu')(fc1)
        dropoutfc1 = Dropout(0.25)(activityfc1)

        fc2 = Dense(128, kernel_initializer='he_normal')(dropoutfc1)
        activityfc2 = Activation('elu')(fc2)
        dropoutfc2 = Dropout(0.25)(activityfc2)

        fc3 = Dense(32, kernel_initializer='he_normal')(dropoutfc2)
        activityfc3 = Activation('elu')(fc3)
        dropoutfc3 = Dropout(0.25)(activityfc3)

        fc4 = Dense(10, kernel_initializer='he_normal')(dropoutfc3)
        activityfc3 = Activation('elu')(fc4)
        dropoutfc3 = Dropout(0.5)(activityfc3)

        fc4 = Dense(num_classes, kernel_initializer='he_normal')(dropoutfc3)
        activityfc4 = Activation('softmax')(fc4)

        Merged = Model(outputs=[activityfc4], inputs=[Input_layer])

        # Merged.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        Merged.summary()
        return Merged
def autoencoder(input_shape):
        fs = 100
        Input_layer = Input(shape=(input_shape), name='inputlayer')
        # Encoder
        # conv1 = Conv1D(1, kernel_size=fs,strides=3, padding='same', input_shape=input_shape, name='fConv0')(Input_layer)
        # conv1 = ReLU()(conv1)
        #maxpooling1 = MaxPooling1D(pool_size=3)(conv1)

        conv1 = Conv1D(8, kernel_size=3, padding='same', input_shape=input_shape,name='fConv1')(Input_layer)
        conv1 = ReLU()(conv1)
        maxpooling1 = MaxPooling1D(pool_size=2)(conv1)

        conv1 = Conv1D(32, kernel_size=5, padding='same')(maxpooling1)
        conv1 = ReLU()(conv1)
        conv1 = BatchNormalization(momentum=0.99)(conv1)
        maxpooling1 = MaxPooling1D(pool_size=2)(conv1)

        #drop1 = Dropout(0.5)(maxpooling1)

        conv2 = Conv1D(filters=16, kernel_size=3*3, padding='same', name='fConv3')(maxpooling1)
        conv2 = ReLU()(conv2)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        maxpooling = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(filters=64, kernel_size=11, padding='same', name='fConv4')(maxpooling)
        conv3 = ReLU()(conv3)
        conv4 = Conv1D(filters=128, kernel_size=13, padding='same', name='fConv5')(conv3)
        conv4 = ReLU()(conv4)
        maxpooling = MaxPooling1D(pool_size=2)(conv4)

        conv6 = Conv1D(filters=32, kernel_size=3, padding='same', name='fConv6')(maxpooling)
        conv6 = ReLU()(conv6)
        conv7 = Conv1D(filters=1, kernel_size=7, padding='same', name='fConv7')(conv6)
        conv7 = ReLU()(conv7)
        maxpooling = MaxPooling1D(pool_size=2)(conv7)

        # Decoder
        decoder=Conv1D(filters=1, kernel_size=7 ,padding='same', name='deConv1')(maxpooling)
        decoder =ReLU()(decoder)
        decoder = Conv1D(filters=32, kernel_size=3, padding='same', name='deConv2')(decoder)
        decoder = ReLU()(decoder)
        decoder =UpSampling1D(2)(decoder)

        decoder = Conv1D(filters=64, kernel_size=11, padding='same', name='deConv3')(decoder)
        decoder = ReLU()(decoder)
        decoder = Conv1D(filters=128, kernel_size=13, padding='same', name='deConv4')(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling1D(2)(decoder)

        decoder = Conv1D(filters=16, kernel_size=3, padding='same', name='deConv5')(decoder)
        decoder = ReLU()(decoder)
        decoder = Conv1D(filters=32, kernel_size=5, padding='same', name='deConv6')(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling1D(2)(decoder)

        decoder = Conv1D(filters=32, kernel_size=3, padding='same', name='deConv8')(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling1D(2)(decoder)
        decoder = Conv1D(filters=8, kernel_size=3, padding='same', name='deConv9')(decoder)
        decoder = ReLU()(decoder)
        #decoder = UpSampling1D(3)(decoder)
        #decoder = Conv1D(filters=32, kernel_size=3, padding='same', name='deConv10')(decoder)
        #decoder = Conv1D(filters=1, kernel_size=1, padding='same', name='deConv0')(decoder)
        #decoder = ReLU()(decoder)
        #decoder = BatchNormalization(momentum=0.99)(decoder)
        #decoder = ReLU()(decoder)
        #decoder = UpSampling1D(2)(decoder)
        #decoder=GlobalAveragePooling1D()(decoder)
        decoder=Flatten()(decoder)
        #decoder = Dense(2000, activation='relu')(decoder)
        #decoder=Dense(6000,activation='sigmoid')(decoder)
        decoder = Dense(6000, activation='linear')(decoder)
        # encoder=Flatten()(Input_layer)
        # encoder=Dense(1000,activation='relu')(encoder)
        # encoder=Dense(200,activation='relu')(encoder)
        # encoder=Dense(40,activation='relu')(encoder)
        #
        # decoder=Dense(200,activation='relu')(encoder)
        # decoder=Dense(1000,activation='relu')(decoder)
        # decoder = Dense(6000, activation='sigmoid')(decoder)
        autoencoder= Model(outputs=[decoder], inputs=[Input_layer])
        autoencoder.summary()
        return autoencoder

def cnn_lstm(input_shape,num_classes):
        fs = 100
        n_step=120
        n_hidden=int(6000/n_step)
        Input_layer = Input(shape=(input_shape), name='inputlayer')
        Input_layer2=Reshape((-1,6000))(Input_layer)
        fc1 = Dense(6000, activation='linear')(Input_layer2)

        slice=Reshape((n_step,n_hidden))(fc1)

        Bilstmlayer1 = (LSTM(64, activation='relu', dropout=0.5, return_sequences=True, name='bLstm1'))(slice)
        Bilstmlayer1 = (LSTM(64, activation='relu', dropout=0.5, return_sequences=False, name='bLstm2'))(Bilstmlayer1)
        fc = Dense(32, activation='relu', name='fc1')(Bilstmlayer1)
        fc = Dense(16, activation='relu', name='fc2')(fc)
        fc = Dense(8, activation='relu', name='fc3')(fc)
        fc = Dense(num_classes, activation='softmax', name='fc4')(fc)

        lstm = Model(outputs=[fc], inputs=[Input_layer])
        lstm.summary()
        return lstm




def fineTuningNet(self):
        inLayer = Input(shape=(self.inputshape ), name='inLayer')
        Layer1, (cShape, fShape) = self.bulid_cnn(inLayer)

        convoutLayer = Dropout(rate=0.5, name='Drop1')(Layer1)

        fc=Dense(256)(convoutLayer)
        ReshapeLayer = Reshape((int(256),1))(convoutLayer)
        Bilstmlayer1= (LSTM(256, activation='relu', dropout=0.5, return_sequences=True,name='bLstm1'))(ReshapeLayer)
        Bilstmlayer1 = (LSTM(256, activation='relu', dropout=0.5, return_sequences=False, name='bLstm1'))(Bilstmlayer1)
        #ReshapeLayer2= Reshape((int(Bilstmlayer1.get_shape()[1]),1))(Bilstmlayer1)
        #Bilstmlayer2=(LSTM(512, activation='relu', dropout=0.5,return_sequences=False, name='bLstm2'))(Bilstmlayer1)
        mergecnnlstm = Concatenate()([fc, Bilstmlayer1])

        mLayer = Dense(1024, activation='relu', name='fc1')(mergecnnlstm)
        mLayer = Dense(512, activation='relu', name='fc2')(mLayer)
        mLayer = Dense(128, activation='relu', name='fc3')(mLayer)
        mLayer = Dense(64, activation='relu', name='fc4')(mLayer)
        mLayer = Dense(32, activation='relu', name='fc5')(mLayer)
        outLayer = Dense(3, activation='softmax', name='outLayer')(mLayer)
        #BilstmoutLayer = Dense(n_classes, activation='softmax', name='outLayer')(Bilstmlayer2)

        network = Model(inLayer, outLayer)
        network.summary()
        return network

def preTrainNet(self):

        inLayer = Input(shape=(self.inputshape), name='inLayer')
        mLayer, (_, _) = self.bulid_cnn(inLayer)
        mLayer=Dense(1024,activation='relu',name='fc1')(mLayer)
        mLayer = Dense(512, activation='relu', name='fc2')(mLayer)
        mLayer = Dense(128, activation='relu', name='fc3')(mLayer)
        mLayer = Dense(64, activation='relu', name='fc4')(mLayer)
        mLayer = Dense(32, activation='relu', name='fc5')(mLayer)
        outLayer = Dense(3, activation='softmax', name='outLayer')(mLayer)
        # outLayer = Dense(n_feats, activation='sigmoid', name='outLayer')(mLayer)

        network = Model(inLayer, outLayer)
        network.summary()
        #network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # network.compile(loss='mean_squared_error', optimizer='adadelta')

        return network


def build_lstm(self):

        self.TimeStep = 480
        self.inputlstm = 3
        OUTPUT_SIZE = 3
        CELL_SIZE = 240

        model = Sequential()

        model.add(LSTM(CELL_SIZE, input_shape=(self.TimeStep, self.inputlstm)))

        # model.add(Dense(240))
        # model.add(Activation('relu'))
        # model.add(Dense(60))
        # model.add(Activation('relu'))

        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))

        model.summary()
        return model

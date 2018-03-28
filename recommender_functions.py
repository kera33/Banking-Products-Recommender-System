# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:50:34 2018

@author: Kera
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense,  Activation
import matplotlib.pyplot as plt


def train_nn(filetraindata='trainingset.csv', filetrainlabels='traininglabels.csv'):
    
    
    
     #this is the recommender model
    def baseline_model(inputlen):
        #create model
        model = Sequential()
        # we use 2 layer 60/60 relu with input original features plus the dimensionality reduction to 3d
        model.add(Dense(60, input_dim=inputlen, init='lecun_uniform')) ##nr samples / input_dim
        model.add(Activation('relu'))
       
        model.add(Dense(60, init='lecun_uniform'))
        model.add(Activation('relu'))
        
        model.add(Dense(12, init='lecun_uniform')) ##nr samples / nr classes
        model.add(Activation('softmax'))
        #compile model
        
        model.compile(loss='hinge',optimizer='adam',metrics=['top_k_categorical_accuracy'])
        return model
    
      #this is used to learn the encoding to 3d of the random forest representation
    def encoder_model(inputlen):
        #create model
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, input_dim=inputlen, init='lecun_uniform', name = 'l1')) ##nr samples / input_dim
        model.add(Activation('relu'))
      
        model.add(Dense(32, init='lecun_uniform', name='l2'))
        model.add(Activation('relu'))
        
        model.add(Dense(16, init='lecun_uniform', name='l3'))
        model.add(Activation('relu'))
        
        model.add(Dense(8, init='lecun_uniform', name='l4'))
        model.add(Activation('relu'))
       
        model.add(Dense(3, init='lecun_uniform', name = 'l5'))
        model.add(Activation('linear'))
        
        model.add(Dense(8, init='lecun_uniform'))
        model.add(Activation('relu'))
        
        model.add(Dense(16, init='lecun_uniform'))
        model.add(Activation('relu'))
        
        model.add(Dense(32, init='lecun_uniform'))
        model.add(Activation('relu'))
        
        model.add(Dense(64, init='lecun_uniform')) 
        model.add(Activation('relu'))
        
        model.add(Dense(inputlen, init='lecun_uniform')) 
        model.add(Activation('linear'))
        #compile model
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model
    
     # this is used at run time to encode to 3d the representation
     # it uses half of the weights previously learned by the autoencoder
    def just_encode_model(inputlen):
        #create model
        model = Sequential()
        #replicates first half of autoencoder up to middle layer 3d
        model.add(Dense(64, input_dim=inputlen, init='lecun_uniform', name='l1')) ##nr samples / input_dim
        model.add(Activation('relu'))
        
        model.add(Dense(32, init='lecun_uniform', name='l2'))
        model.add(Activation('relu'))
        
        model.add(Dense(16, init='lecun_uniform', name = 'l3'))
        model.add(Activation('relu'))
        
        model.add(Dense(8, init='lecun_uniform', name='l4'))
        model.add(Activation('relu'))
        
        model.add(Dense(3, init='lecun_uniform', name='l5'))
        model.add(Activation('relu'))
        
        #compile model
        model.compile(loss='mean_squared_error',optimizer='adam')#top_k_categorical_accuracy(y_true, y_pred, k=7)'])
        return model

    
    # fix random seed for reproductibility
    seed = 7
    np.random.seed(seed)
    
    ##read train data
    X_train = pd.read_csv(filetraindata)
    X_train.drop(['clientid'], axis=1, inplace=True)
    traincolumns = X_train.columns
    pd.DataFrame(traincolumns).to_csv('traincolumns.csv', index=False)
    
    
    ##adding the semi-supervised part
    #train a random tree embedding of 20 trees depth 3 with a max leaf number of 2^3*20
    
    rte = RandomTreesEmbedding(n_estimators=20, max_depth=3, min_samples_split=2, \
                               min_samples_leaf=1, \
                               sparse_output=True, \
                               n_jobs=1, random_state=None )
    
    ##y_train de mai jos
    y_train = pd.read_csv(filetrainlabels)
    #y_train = y_train.iloc[:,1:]
    
    
    #train the embedding supervised with label as last product - investment funds 
    rte.fit(X_train, y_train.iloc[:,-1]  )
    
    #save the Random Trees Embedding to be used for run time
    pickle.dump(rte, open('rte-invfunds','wb'))
    
    #test that it was saved ok
    rte = pickle.load(open('rte-invfunds','rb'))
    
    #transform the float input to sparse vectors with similarity meaning
    X_train_trees = rte.transform(X_train).todense()
    
    #check the distribution of leafs density
    #leafs = pd.DataFrame(X_train_trees.sum(axis=0)).values[0]
    #plt.scatter(range(len(leafs)),leafs)
    
    #compute the length of the embedding, somewhere around 160 but smaller ..
    nrleafs = int(X_train_trees.shape[1])
    
    #train the deep autonecoder to compress the data to a smaller dimension
    encmodel=encoder_model(nrleafs)
    encmodel.fit(X_train_trees, X_train_trees,
              epochs=4,
              batch_size=100)
    
    #save the autoencoder weights
    filename = 'model-encoder-funds.hdf5'
    encmodel.save_weights(filename)

    #encmodel.summary()
    
    ##build just the encode part by loading half of weights
    justencmodel = just_encode_model(nrleafs)
    #justencmodel.summary()
    justencmodel.load_weights(filename, by_name=True)
    justencmodel.save('encoder.hdf5')
    
    #compute the 3d embedding of original data
    enc_x_train = pd.DataFrame(justencmodel.predict(X_train_trees))
    
    #check that the dimensionality reduction is well distributed 
    #if some dimensions are zero, try training a new encoder as it depends on random initialization
    
    #plt.scatter(range(len(enc_x_train)), enc_x_train.iloc[:,0])
    #plt.scatter(range(len(enc_x_train)), enc_x_train.iloc[:,1])
    #plt.scatter(range(len(enc_x_train)), enc_x_train.iloc[:,2])
    #plt.scatter(enc_x_train.iloc[:,0], enc_x_train.iloc[:,1])
   
    ##end semi-supervised

    
    ##new greater X_train 
    X_train = pd.concat([X_train,enc_x_train], axis=1)
   
    #scale the input
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    #save the StandardScaler to be used for run time
    pickle.dump(X_scaler, open('scaler','wb'))
        
    y_train = y_train.values
    
    #train the recommender
    model=baseline_model(X_train.shape[1])
    model.fit(X_train, y_train,
              epochs=10,
              batch_size=100)
                   
    model.save('recommender.hdf5')
    ##0.9608 la rte funds 20 epochs, 0.9469
    ##0.64 final cu hinge simplu
    
    
    
    return "Done"

classifiername='nn'
def test_nn(filetestdata='testset.csv', filetestlabels='testlabels.csv' ):
    
    def create_strings_labels(df, filename):
        ##transform predictions into product list per client
        tdf = pd.DataFrame(df.iloc[:,-1])
        tdf['pred'] = ''
        for col in range(13):
            tdf.loc[df.iloc[:,col]==1,'pred'] =  tdf.loc[df.iloc[:,col]==1,'pred']+","+df.columns[col]
        tdf['pred'] = tdf['pred'].str.slice(1,1000)
        tdf[['clientid', 'pred']].to_csv(filename, index=False)
      

    
    def create_strings_pred(classifiername):
        
        print("strings pred {}".format(classifiername))
        filename = 'results-'+classifiername+'.csv'  
        df = pd.read_csv(filename)  
        origcols = df.columns
        df.columns = range(13)
        
        ##assign products in order of proba
        df['ind'] = df.index
        k=1
        while k<13:
            df['max'] = df.iloc[:,1:13].max(axis=1)
            for i in range(12):
                df.loc[:,1+i] = df.loc[:,1+i]-df.loc[:,'max']
            
            
            df['max'+str(k)] =0
            for i in range(12):
                indexes = df.loc[((df['max'+str(k)] ==0) & (df.loc[:,1+i]==0)),'ind']
                df.loc[indexes,'max'+str(k)] = df.loc[indexes,'max'+str(k)]+i+1
                df.loc[indexes,1+i] = df.loc[indexes,1+i]-100*(df.loc[indexes,1+i]==0).astype(int)
            
            k=k+1
        
        ##change it back to strings
        k=1
        while k<13:
             df['max'+str(k)]=origcols[ df['max'+str(k)].astype(int)]
             k=k+1    
              
        ##get prediction top k , k=5 default
        maxk=5
        df['pred'] = df['max1']
        k=2
        while k<=maxk:
           df['pred']= df['pred']+','+df['max'+str(k)] 
           k=k+1
        
        df['clientid'] = df.iloc[:,0]
        df[['clientid', 'pred']].to_csv('pred-strings-'+classifiername+'.csv', index=False)
        return "Done"

    def avgprecision(actual,predicted,maxnum=5):
        if len(predicted)>maxnum:
            predicted = predicted[:maxnum]
        score = 0.0
        hits = 0.0
        for index, prediction in enumerate(predicted):
            if prediction in actual and prediction not in predicted[:index]:
                hits +=1
                score += hits/(index+1.0)
        if not actual:
            return 0
        return score/min(len(actual),maxnum)

    def meanavgprecision2(actual, predicted, maxnum=5):
        """Mean average precision only for actual not blank"""
        print(len([avgprecision(a,p,maxnum) for a, p in zip(actual, predicted) if a!=['']]))
        return np.mean([avgprecision(a,p,maxnum) for a, p in zip(actual, predicted) if a!=['']])

    
    def score_map5(predfile, testfile):

        test = pd.read_csv(testfile).fillna('')
        testcols = test.columns
        pred = pd.read_csv(predfile).fillna('')
        
        testvalues = test['pred'].map(lambda a: a.split(',')).tolist()
        predvalues = pred['pred'].map(lambda a: a.split(',')).tolist()
        
        return meanavgprecision2(testvalues,predvalues,5)
    
    #load good columns saved by the trainer
    traincolumns = pd.read_csv('traincolumns.csv')
    traincolumns = traincolumns.iloc[:,0].values
    #load test data
    fileX_cv = pd.read_csv(filetestdata)
    ##necessary in case the column order in the testdata is different than traindata
    X_cv = fileX_cv.loc[:,traincolumns]
    
    #load test labels
    y_cv = pd.read_csv(filetestlabels)
    ##save the ground truth in a string format per client
    testfile = 'labels-strings.csv'
    create_strings_labels(y_cv, testfile)
    #remember the client
    cont = y_cv['clientid']
    #clean for training
    y_cv.drop(['clientid'], axis=1, inplace=True)
    
    #load rte already trained
    rte = pickle.load(open('rte-invfunds','rb'))
    #apply random trees embedding
    x_cv_trees = rte.transform(X_cv).todense()
    
    #load encoder trained
    justencmodel = load_model('encoder.hdf5')
    #apply encoder
    enc_x_cv = pd.DataFrame(justencmodel.predict(x_cv_trees))
    #enrich original data with the 3d view
    X_cv = pd.concat([X_cv, enc_x_cv], axis=1)
    
    #load and apply same scaler used for training
    X_scaler = pickle.load(open('scaler','rb'))
    X_cv = X_scaler.transform(X_cv)
    
    #load recommender
    model = load_model('recommender.hdf5')
    y_pred = model.predict_proba(X_cv, verbose=2)
     
    y_pred = pd.DataFrame(y_pred)
    #pcp1 = pd.DataFrame(y_pred+y_pred2)
    y_pred.columns = y_cv.columns
    y_pred = pd.concat([cont,y_pred],axis=1)
    y_pred.to_csv("results-nn.csv",index=False)
    
    #transform predictions to product strings per client
    create_strings_pred('nn')
    
    #compute MAP@5 between prediction and truth
    score = score_map5('pred-strings-nn.csv', testfile)
    print(score)
    
    
   
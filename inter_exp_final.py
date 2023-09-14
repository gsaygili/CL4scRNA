
import numpy as np
import pandas as pd
import time as tm
import natsort
import random
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import clone
import glob
import os
#os.environ["R_HOME"] = r"C:/Program Files/R/R-4.2.1"
import rpy2.robjects as robjects
from xgboost import XGBClassifier
import plot_save_results as plts
warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, roc_auc_score, precision_score, recall_score


class benchmark_inter():
    '''
    This class loads the benchmark dataset
    '''
    def __init__(self, DataPath, LabelsPath, CV_RDataPath):
        self.DataPath = DataPath
        self.LabelsPath = LabelsPath
        self.RPath = CV_RDataPath
        self.data = None
        self.labels = None
        self.eliminated_data = None
        self.eliminated_labels = None
        self.y_pred = None
        self.labelencoder = None
        self.truelab = None
        self.trtime = 0
        self.tetime = 0

    def normalize(self,arr):
        return np.log1p(arr)
            
    def read_data(self):
        # read the data
        data = pd.read_csv(self.DataPath,index_col=0,sep=',').to_numpy()
        self.labels = pd.read_csv(self.LabelsPath, header=0,
                                  index_col=None, sep=',').to_numpy()
        robjects.r['load'](self.RPath)        
        #le = preprocessing.LabelEncoder()
        self.data = self.normalize(data)
        #self.labels = le.fit_transform(labels)
        #self.labelencoder = le
        
        #to only keep joint classes ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']
        eliminated_labels = self.labels[[(self.labels[i] == ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']).any() for i in range(len(self.labels))]]
        self.eliminated_data =  self.data[[(self.labels[i] == ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']).any() for i in range(len(self.labels))]]
               
        le = preprocessing.LabelEncoder()
        self.eliminated_labels = le.fit_transform(eliminated_labels)
        self.labelencoder = le
        
        print("data shape: ", np.shape(self.data))
        print("labels shape: ", np.shape(self.labels))
        
        
    def remove_class(self, label_list, data, labels):
        
        y_train = labels[[(labels[i] == label_list).any() for i in range(len(labels))]]
        X_tr =  data[[(labels[i] == label_list).any() for i in range(len(labels))]]
        y_tr = self.labelencoder.transform(y_train) 
        
        return X_tr, y_tr
    
    def split_data(self, savePath, dataName):
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        xls = pd.ExcelFile('.../sc-RNAseq/benchmark/Inter-dataset/PbmcBench/Statisitics.xlsx')
        df1 = pd.read_excel(xls, 'Sheet2')
        #We have total 7 datasets in Pbmc
        loop = 7
        
        if dataName == "X10v3.pbmc1":
            loop = 6
        # To extract the first data as our test set, we make use of the CVRdata.    
        train_ind = np.array(robjects.r['Train_Idx']) 

        X_train = self.data[train_ind][0,:,:]
        y_train = self.labels[train_ind][0]
        print(np.shape(X_train))
        print(np.shape(y_train))
        
        #to only keep joint classes ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']
        label_list = ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']
        X_tr, y_tr = self.remove_class(label_list, X_train, y_train)

        #First dataset (X10v2 dataset) saved as Fold0
        np.save(savePath+"X_Fold_0_"+ dataName, X_tr)
        np.save(savePath+"y_Fold_0_"+ dataName, y_tr)
        
        #Finds the index of related dataset -- it returns 8 for X10v2 dataset.
        indx = int(np.where(df1['Train'] == dataName)[0])
        
        # iterate through all datasets
        for i in range(loop):
            ind = indx + i
            Test_name = df1['Test'][ind]
            print('Test_name : ', Test_name)
            test_idx = [int(i) for i in (df1['Idx for testing'][ind].split(':'))]
            
            X_data = self.data[test_idx[0]:test_idx[1]]
            y_data = self.labels[test_idx[0]:test_idx[1]]
            
            #to only keep joint classes ['B cell','CD14+ monocyte','CD4+ T cell','Cytotoxic T cell']
            X_test, y_test = self.remove_class(label_list, X_data, y_data)
            
            np.save(savePath+"X_Fold_"+str(i+1)+"_"+ Test_name, X_test)
            np.save(savePath+"y_Fold_"+str(i+1)+"_"+ Test_name, y_test)

    def fivefoldcv_linear_svm(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        
        #Randomly sort the data order!!
        c = list(zip(files, fileslab))
        random.Random(4).shuffle(c) #Random(4) gives same random order each time for the comparison with other models
        files, fileslab  = zip(*c)
        
        num_folds = len(files)
        print("linear svm learning starts...")
        y_pred = []
        y_true = []
        x_train = []
        x_test = []
        outFold = path2folds + "linSVM_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            print("test fold: " + str(i))
            frst = True
            for j in range(num_folds): # training folds
                if j != i:   
                    print("file is been processed: "+files[j])
                    if frst:
                        x_train = np.load(files[j], allow_pickle=True)
                        y_train = np.load(fileslab[j], allow_pickle=True)
                        frst = False
                    else:
                        x_train = np.append(x_train, np.load(files[j], allow_pickle=True), axis=0)
                        y_train = np.append(y_train, np.load(fileslab[j], allow_pickle=True), axis=0)
                    print("X_train is read")
                    print("y_train is read")
            print(x_train.shape)
            print(y_train.shape)
            start=tm.time()
            clf.fit(x_train, y_train)
            tr_time += tm.time()-start
            x_test = np.load(files[i], allow_pickle=True)
            if i==0: 
                y_true = np.load(fileslab[i], allow_pickle=True)
                start=tm.time()
                y_pred = clf.predict(x_test)
                ts_time += tm.time()-start
            else:
                y_true = np.append(y_true, np.load(fileslab[i], allow_pickle=True), axis=0)    
                start=tm.time()
                y_pred = np.append(y_pred, clf.predict(x_test), axis=0)
                ts_time += tm.time()-start
            print(y_pred.shape)
            
            outFold2 = outFold + 'Fold_'+ str(i) +'/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold2)
                         
            self.y_pred = y_pred
            self.truelab = y_true
            self.trtime = tr_time
            self.tetime = ts_time
            plts_obj = plts.plot_save(self, outFold2)
            plts_obj.save_results()
            # save with original labels
            outFold3 = path2folds + "linSVM_results/with_original_labels/"+'Fold_'+ str(i) + '/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold3)
            self.y_pred = self.labelencoder.inverse_transform(y_pred.astype(dtype=int))
            self.truelab = self.labelencoder.inverse_transform(y_true.astype(dtype=int))
            plts_obj = plts.plot_save(self, outFold3)
            plts_obj.save_results()

    def fivefoldcv_online_xgboost(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse = False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        
        #Randomly sort the data order!!
        c = list(zip(files, fileslab))
        random.Random(4).shuffle(c) #Random(4) gives same random order each time for the comparison with other models
        files, fileslab  = zip(*c)
        
        num_folds = len(files)
        print("online learning starts...")

        f1_scores = []
        outFold = path2folds + "xgboost_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
            outFold2 = outFold +'Fold_'+ str(i) +'/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold2)  
            print("test fold: " + str(i))
            for j in range(num_folds): # training folds
                if j != i:   
                    print("file is been processed: "+files[j])
                    x_train = np.load(files[j], allow_pickle=True)
                    print("X_train is read")
                    y_train = np.load(fileslab[j], allow_pickle=True)
                    print(np.unique(y_train))
                    print("y_train is read")
                    start=tm.time()
         
                    if frst == 0:
                        print("first training")
                        clf = clf.fit(x_train, y_train)
                        frst = 1
                    else:
                        print("online training - after first training")
                        clf = clf.fit(x_train, y_train, xgb_model=clf)
                        
                    tr_time += tm.time()-start
                    x_test = np.load(files[i], allow_pickle=True)
                    y_true = np.load(fileslab[i], allow_pickle=True)                    
                    start=tm.time()
                    y_pred = clf.predict(x_test)
                    ts_time += tm.time()-start
                    
                    outFold3 = outFold2 + str(j)
                    if not os.path.exists(outFold3):
                        os.makedirs(outFold3)
                        
                    self.y_pred = y_pred
                    print("y_pred shape: " + str(self.y_pred.shape))
                    self.truelab = y_true
                    print("y_true shape: " + str(self.truelab.shape))
                    self.trtime = tr_time
                    self.tetime = ts_time
                    plts_obj = plts.plot_save(self, outFold3)
                    plts_obj.save_results()  
                    
                    #f1_scores.append(f1_score(y_true, y_pred, average="macro"))
                    
                    # save with original labels
                    outFold4 = path2folds + "xgboost_results/with_original_labels/"+'Fold_'+ str(i) +'/'+str(j) 
                    if not os.path.exists(outFold4):
                        os.makedirs(outFold4)
                    self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
                    self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
                    plts_obj = plts.plot_save(self, outFold4)
                    plts_obj.save_results()
                np.save(outFold2 +"f1_scores_fold_"+str(i), f1_scores)
                      
            
    def fivefoldcv_online_scikitlearn(self, path2folds= None, Classifier=None, outFold=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        
        #Randomly sort the data order!!
        c = list(zip(files, fileslab))
        random.Random(4).shuffle(c) #Random(4) gives same random order each time for the comparison with other models
        files, fileslab  = zip(*c)
        
        num_folds = len(files)
        print("online learning with sgd starts...")
        y_pred = []
        f1_scores = []
        outFold1 = path2folds + outFold
        print("outFold1: " + outFold1)
        if not os.path.exists(outFold1):
            os.makedirs(outFold1)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            #frst = 0
            outFold2 = outFold1 +'Fold_'+ str(i) +'/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold2)  
            print("test fold: " + str(i))
            for j in range(num_folds): # training folds
                if j != i:   
                    print("file is been processed: "+files[j])
                    x_train = np.load(files[j], allow_pickle=True)
                    print("X_train is read")
                    y_train = np.load(fileslab[j], allow_pickle=True)
                    print(np.unique(y_train))
                    print("y_train is read")
                    start=tm.time()
                    clf = clf.partial_fit(x_train, y_train, classes=np.unique(self.eliminated_labels))
                    tr_time += tm.time()-start
                    
                    x_test = np.load(files[i], allow_pickle=True)
                    y_true = np.load(fileslab[i], allow_pickle=True)                    
                    start=tm.time()
                    y_pred = clf.predict(x_test)
                    ts_time += tm.time()-start
                    
                    outFold3 = outFold2 + str(j)
                    if not os.path.exists(outFold3):
                        os.makedirs(outFold3)                       
                    self.y_pred = y_pred
                    print("y_pred shape: " + str(self.y_pred.shape))
                    self.truelab = y_true
                    print("y_true shape: " + str(self.truelab.shape))
                    self.trtime = tr_time
                    self.tetime = ts_time
                    plts_obj = plts.plot_save(self, outFold3)
                    plts_obj.save_results()                    
                    f1_scores.append(f1_score(y_true, y_pred, average="macro"))
                    # save with original labels
                    outFold4 = path2folds + outFold + "with_original_labels/" +'Fold_'+ str(i) +'/' + str(j)
                    if not os.path.exists(outFold4):
                        os.makedirs(outFold4)
                    self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
                    self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
                    plts_obj = plts.plot_save(self, outFold4)
                    plts_obj.save_results()
                #np.save(outFold2 +"f1_scores_fold_"+str(i), f1_scores)
                
    def fivefoldcv_online_lightgbm(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse = False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        
        #Randomly sort the data order!!
        c = list(zip(files, fileslab))
        random.Random(4).shuffle(c) #Random(4) gives same random order each time for the comparison with other models
        files, fileslab  = zip(*c)
        
        num_folds = len(files)
        print("online learning with lightgbm starts...")

        f1_scores = []
        outFold = path2folds + "lightgbm_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
            outFold2 = outFold +'Fold_'+ str(i) +'/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold2)  
            print("test fold: " + str(i))
            for j in range(num_folds): # training folds
                if j != i:   
                    print("file is been processed: "+files[j])
                    x_train = np.load(files[j], allow_pickle=True)
                    print("X_train is read")
                    y_train = np.load(fileslab[j], allow_pickle=True)
                    print(np.unique(y_train))
                    print("y_train is read")
                    start=tm.time()
         
                    if frst == 0:
                        print("first training")
                        clf = clf.fit(x_train, y_train)
                        frst = 1
                    else:
                        print("online training - after first training")
                        clf = clf.fit(x_train, y_train, init_model=clf)
                        
                    tr_time += tm.time()-start
                    x_test = np.load(files[i], allow_pickle=True)
                    y_true = np.load(fileslab[i], allow_pickle=True)                    
                    start=tm.time()
                    y_pred = clf.predict(x_test)
                    ts_time += tm.time()-start
                    
                    outFold3 = outFold2 + str(j)
                    if not os.path.exists(outFold3):
                        os.makedirs(outFold3)
                        
                    self.y_pred = y_pred
                    print("y_pred shape: " + str(self.y_pred.shape))
                    self.truelab = y_true
                    print("y_true shape: " + str(self.truelab.shape))
                    self.trtime = tr_time
                    self.tetime = ts_time
                    plts_obj = plts.plot_save(self, outFold3)
                    plts_obj.save_results()  
                    
                    #f1_scores.append(f1_score(y_true, y_pred, average="macro"))
                    
                    # save with original labels
                    outFold4 = path2folds + "lightgbm_results/with_original_labels/"+'Fold_'+ str(i) +'/'+str(j) 
                    if not os.path.exists(outFold4):
                        os.makedirs(outFold4)
                    self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
                    self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
                    plts_obj = plts.plot_save(self, outFold4)
                    plts_obj.save_results()
                np.save(outFold2 +"f1_scores_fold_"+str(i), f1_scores)
        
    def fivefoldcv_online_catboost(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
       	files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        
        #Randomly sort the data order!!
        c = list(zip(files, fileslab))
        random.Random(4).shuffle(c) #Random(4) gives same random order each time for the comparison with other models
        files, fileslab  = zip(*c)
        
        num_folds = len(files)
        print("online learning with catboost starts...")
        y_pred = []
        f1_scores = []
        outFold = path2folds + "catboost_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
            outFold2 = outFold +'Fold_'+ str(i) +'/'
            if not os.path.exists(outFold2):
                os.makedirs(outFold2)
            print("test fold: " + str(i))
            for j in range(num_folds): # training folds
                if j != i:   
                    print("file is been processed: "+files[j])
                    x_train = np.load(files[j], allow_pickle=True)
                    print("X_train is read")
                    y_train = np.load(fileslab[j], allow_pickle=True)
                    print(np.unique(y_train))
                    print("y_train is read")
                    start=tm.time()
                    if frst == 0:
                        print("first training")
                        clf = clf.fit(x_train, y_train)
                        frst = 1
                    else:
                        print("online training - after first training")
                        clf = clf.fit(x_train, y_train, init_model=clf)
                    tr_time += tm.time()-start
                    x_test = np.load(files[i], allow_pickle=True)
                    y_true = np.load(fileslab[i], allow_pickle=True)

                    start=tm.time()
                    y_pred = clf.predict(x_test)
                    ts_time += tm.time()-start
                    outFold3 = outFold2 + str(j)
                    if not os.path.exists(outFold3):
                        os.makedirs(outFold3)
                    
                    self.y_pred = y_pred
                    print("y_pred shape: " + str(self.y_pred.shape))
                    self.truelab = y_true
                    print("y_true shape: " + str(self.truelab.shape))
                    self.trtime = tr_time
                    self.tetime = ts_time
                    plts_obj = plts.plot_save(self, outFold3)
                    plts_obj.save_results()
                    f1_scores.append(f1_score(y_true, y_pred, average="macro"))
                    # save with original labels
                    outFold4 = path2folds + "catboost_results/with_original_labels/" + 'Fold_'+ str(i) +'/' + str(j)
                    if not os.path.exists(outFold4):
                        os.makedirs(outFold4)
                    self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
                    self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
                    plts_obj = plts.plot_save(self, outFold)
                    plts_obj.save_results()
                np.save(outFold2 +"f1_scores_fold_"+str(i), f1_scores)

        
            
if __name__ == "__main__":
    benchPath = ".../sc-RNAseq/benchmark/Inter-dataset/PbmcBench/"

    dataPaths = ["10Xv2/"]

    dNames = ["10Xv2_pbmc1.csv"]

    dataName = ["X10v2.pbmc1"]

    # find the RData file name
    for i, datPath in enumerate(dataPaths):
        
        dName = dNames[i]
        datName = dataName[i]
        rName = glob.glob(benchPath+datPath+"/*.RData")[0]
        obj = benchmark_inter(DataPath=benchPath+datPath+dName, 
                              LabelsPath=benchPath+datPath+dName[:-4]+"Labels.csv",
                              CV_RDataPath=rName)
        
        # read data
        print("reading the data...")
        obj.read_data()
        # create folds if has not been done
        print("splitting the data...")
        #obj.split_data(savePath=benchPath+datPath+"folds/", dataName = datName)

        xgb = XGBClassifier()
        linsvm = LinearSVC()
        ctb = CatBoostClassifier()
        lgb = LGBMClassifier()
        
        # scikit learn
        pac = PassiveAggressiveClassifier()
        sgd = SGDClassifier()
        per = Perceptron()

        print("Linear SVM experiment")
        obj.fivefoldcv_linear_svm(path2folds= benchPath+datPath+"folds/", Classifier=linsvm) 
        print("XGBoost experiment")
        obj.fivefoldcv_online_xgboost(path2folds= benchPath+datPath+"folds/", Classifier=xgb)
        print("LightGBM experiment")
        obj.fivefoldcv_online_lightgbm(path2folds= benchPath+datPath+"folds/", Classifier=lgb)
        print("SGD Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=SGDClassifier(loss='log'), outFold="sgd_log_results/")
        print("passiveaggressive Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=pac, outFold="passiveaggresive_results/")
        print("perceptron Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=per, outFold="perceptron_results/")
        #obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=sgd, outFold="sgd_results/")
        #print("CatBoost Classifier experiment")
        #obj.fivefoldcv_online_catboost(path2folds= benchPath+datPath+"folds/", Classifier=ctb)
        # print("SGD Classifier experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=SGDClassifier(penalty = 'l2'), outFold="sgd_results_pen_l2/")
        # print("SGD Classifier experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=SGDClassifier(penalty = 'l1'), outFold="sgd_results_pen_l1/")
        # print("SGD Classifier experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=SGDClassifier(penalty = 'elasticnet'), outFold="sgd_results_pen_elasticnet/")
        # print("Perceptron experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=Perceptron(penalty = 'l2'), outFold="perceptron_results_pen_l2/")
        # print("Perceptron experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=Perceptron(penalty = 'l1'), outFold="perceptron_results_pen_l1/")
        # print("Perceptron experiment")
        # obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds/", Classifier=Perceptron(penalty = 'elasticnet'), outFold="perceptron_results_pen_elasticnet/")

        
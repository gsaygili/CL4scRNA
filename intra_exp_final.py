
import numpy as np
import pandas as pd
import time as tm
import natsort
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
from sklearn.neighbors import KNeighborsClassifier

class benchmark_intra():
	
    def __init__(self, DataPath, LabelsPath, CV_RDataPath, k=10):
        self.DataPath = DataPath
        self.LabelsPath = LabelsPath
        self.RPath = CV_RDataPath
        self.data = None
        self.labels = None
        self.k = k    	

    def normalize(self,arr):
        return np.log1p(arr)

    def read_data(self):
        # read the data
        self.data = pd.read_csv(self.DataPath,index_col=0,sep=',').to_numpy()
        self.labels = pd.read_csv(self.LabelsPath, header=0,
                                  index_col=None, sep=',').to_numpy()
        robjects.r['load'](self.RPath)
        self.nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
        tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
        col = np.array(robjects.r['col_Index'], dtype = 'int')
        col = col - 1 
        #self.test_ind = np.array(robjects.r['Test_Idx'])
        #self.train_ind = np.array(robjects.r['Train_Idx'])
        data = pd.read_csv(self.DataPath,index_col=0,sep=',')
        labels = pd.read_csv(self.LabelsPath, header=0,
                                  index_col=None, sep=',', usecols = col)
        labels = labels.iloc[tokeep]
        data = data.iloc[tokeep]
        if labels.shape[1]>1:
            labels = labels[:,-1]
            print(labels.shape)
        labels = labels.to_numpy()
        le = preprocessing.LabelEncoder()
        #self.data = self.normalize(data.to_numpy())
	#for the HLCA data, no need to normalization
        self.data = data.to_numpy()
        self.labels = le.fit_transform(labels)
        print(self.labels)
        self.labelencoder = le

    def apply_stratifiedKfold(self, savePath):
        skf = StratifiedKFold(n_splits=self.k)
        skf.get_n_splits(self.data, self.labels)
        for i, (train_index, test_index) in enumerate(skf.split(self.data, self.labels)):
            print("fold index: " + str(i))
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            test = self.data[test_index]
            y_test = self.labels[test_index]

            # save train test sets into a folder
            np.save(savePath+"X_fold_"+str(i), test)
            np.save(savePath+"y_fold_"+str(i), y_test)
            
    def fivefoldcv_online_xgboost(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        num_folds = len(files)
        print("online learning starts...")
        y_pred = []
        outFold = path2folds + "xgboost_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
            print("test fold: " + str(i))
            for j in range(num_folds): # training folds
                if j != i:
                    print("file is been processed: " + files[j])
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
            if i == 0:
                y_true = np.load(fileslab[i], allow_pickle=True)
            else:
                y_true = np.append(y_true, np.load(fileslab[i], allow_pickle=True), axis=0)
            start=tm.time()
            if i==0:
                y_pred = clf.predict(x_test)
            else:
                y_pred = np.append(y_pred, clf.predict(x_test), axis=0)
            ts_time += tm.time()-start

        self.y_pred = y_pred
        print("y_pred shape: " + str(self.y_pred.shape))
        self.truelab = y_true
        print("y_true shape: " + str(self.truelab.shape))
        acc_score = accuracy_score(self.truelab, self.y_pred)
        print("accuracy score: " + str(acc_score))
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        # save with original labels
        outFold = path2folds + "xgboost_results/with_original_labels/"
        self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
        self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
    

    def fivefoldcv_online_lightgbm(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        num_folds = len(files)
        print("online learning with lightgbm starts...")
        y_pred = []
        outFold = path2folds + "lightgbm_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
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
            if i == 0:
                y_true = np.load(fileslab[i], allow_pickle=True)
            else:
                y_true = np.append(y_true, np.load(fileslab[i], allow_pickle=True), axis=0)
            start=tm.time()
            if i==0:
                y_pred = clf.predict(x_test)
            else:
                y_pred = np.append(y_pred, clf.predict(x_test), axis=0)
            ts_time += tm.time()-start

        self.y_pred = y_pred
        print("y_pred shape: " + str(self.y_pred.shape))
        self.truelab = y_true
        print("y_true shape: " + str(self.truelab.shape))
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        # save with original labels
        outFold = path2folds + "lightgbm_results/with_original_labels/"
        self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
        self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        
    def fivefoldcv_online_catboost(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        num_folds = len(files)
        print("online learning with catboost starts...")
        y_pred = []
        outFold = path2folds + "catboost_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
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
            if i == 0:
                y_true = np.load(fileslab[i], allow_pickle=True)
            else:
                y_true = np.append(y_true, np.load(fileslab[i], allow_pickle=True), axis=0)
            start=tm.time()
            if i==0:
                y_pred = clf.predict(x_test)
            else:
                y_pred = np.append(y_pred, clf.predict(x_test), axis=0)
            ts_time += tm.time()-start

        self.y_pred = y_pred
        print("y_pred shape: " + str(self.y_pred.shape))
        self.truelab = y_true
        print("y_true shape: " + str(self.truelab.shape))
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        # save with original labels
        outFold = path2folds + "catboost_results/with_original_labels/"
        self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
        self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()

    def fivefoldcv_online_scikitlearn(self, path2folds= None, Classifier=None, outFold=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        num_folds = len(files)
        print("online learning with catboost starts...")
        y_pred = []
        outFold1 = path2folds + outFold
        if not os.path.exists(outFold1):
            os.makedirs(outFold1)
        for i in range(num_folds): # test fold
            clf = clone(Classifier)
            frst = 0
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
                    clf = clf.partial_fit(x_train, y_train, classes=np.unique(self.labels))
                    tr_time += tm.time()-start
            x_test = np.load(files[i], allow_pickle=True)
            if i == 0:
                y_true = np.load(fileslab[i], allow_pickle=True)
            else:
                y_true = np.append(y_true, np.load(fileslab[i], allow_pickle=True), axis=0)
            start=tm.time()
            if i==0:
                y_pred = clf.predict(x_test)
            else:
                y_pred = np.append(y_pred, clf.predict(x_test), axis=0)
            ts_time += tm.time()-start

        self.y_pred = y_pred
        print("y_pred shape: " + str(self.y_pred.shape))
        self.truelab = y_true
        print("y_true shape: " + str(self.truelab.shape))
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold1)
        plts_obj.save_results()
        # save with original labels
        outFold2 = outFold1 + "with_original_labels/"
        self.y_pred = self.labelencoder.inverse_transform(np.array(y_pred))
        self.truelab = self.labelencoder.inverse_transform(np.array(y_true))
        plts_obj = plts.plot_save(self, outFold2)
        plts_obj.save_results()    
    def fivefoldcv_linear_svm(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
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
                    else:
                        x_train = np.append(x_train, np.load(files[j], allow_pickle=True), axis=0)
                        y_train = np.append(y_train, np.load(fileslab[j], allow_pickle=True), axis=0)
                    print("X_train is read")
                    print("y_train is read")
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

        self.y_pred = y_pred
        self.truelab = y_true
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        # save with original labels
        outFold = path2folds + "linSVM_results/with_original_labels/"
        print(y_pred)
        self.y_pred = self.labelencoder.inverse_transform(y_pred.astype(dtype=int))
        self.truelab = self.labelencoder.inverse_transform(y_true.astype(dtype=int))
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()

    def fivefoldcv_knn(self, path2folds= None, Classifier=None):
        tr_time = 0
        ts_time = 0
        files = glob.glob(path2folds + "/X_fold_*")
        files = natsort.natsorted(files, reverse=False)
        fileslab = glob.glob(path2folds + "/y_fold_*")
        fileslab = natsort.natsorted(fileslab , reverse=False)
        num_folds = len(files)
        print("knn learning starts...")
        y_pred = []
        y_true = []
        x_train = []
        x_test = []
        outFold = path2folds + "KNN_results/"
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        for i in range(1,num_folds): # test fold
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

        self.y_pred = y_pred
        self.truelab = y_true
        self.trtime = tr_time
        self.tetime = ts_time
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()
        # save with original labels
        outFold = path2folds + "KNN_results/with_original_labels/"
        print(y_pred)
        self.y_pred = self.labelencoder.inverse_transform(y_pred.astype(dtype=int))
        self.truelab = self.labelencoder.inverse_transform(y_true.astype(dtype=int))
        plts_obj = plts.plot_save(self, outFold)
        plts_obj.save_results()

        
if __name__ == "__main__":
    benchPath = ".../benchmark/Intra-dataset/"

    dataPaths = ["AMB/", "CellBench/10x_5cl/", "CellBench/CelSeq2_5cl/", \
    "Pancreatic_data/Baron Human/", "Pancreatic_data/Baron Mouse/", \
    "Pancreatic_data/Muraro/", "Pancreatic_data/Segerstolpe/", \
    "Pancreatic_data/Xin/", "TM/", "Zheng_68K/", "Zheng_sorted/"]
    #dataPaths = ["HLCA/"]
    #dataPaths = ["HLCA/Emb/"]
    #dataPaths = ["HLCA/", "EQTL/", "AMB/"] 
    dNames = ["Filtered_mouse_allen_brain_data.csv", "10x_5cl_data.csv", \
    "CelSeq2_5cl_data.csv", "Filtered_Baron_HumanPancreas_data.csv", \
    "Filtered_MousePancreas_data.csv", "Filtered_Muraro_HumanPancreas_data.csv", \
    "Filtered_Segerstolpe_HumanPancreas_data.csv", "Filtered_Xin_HumanPancreas_data.csv", \
    "Filtered_TM_data.csv", "Filtered_68K_PBMC_data.csv", "Filtered_DownSampled_SortedPBMC_data.csv"]
    #dNames = ["data_HLCA.csv"] 
    #dNames = ["data_emb_HLCA.csv"] 
    #dNames = ["data_HLCA.csv", "Data_EQTL.csv", "Filtered_mouse_allen_brain_data.csv"]
 

    # find the RData file name
    for i, datPath in enumerate(dataPaths):
        print("index: "+str(i)+" data path: " + str(datPath))
        dName = dNames[i]
        rName = glob.glob(benchPath+datPath+"/*.RData")[0]
        Kfold = 5
        obj = benchmark_intra(DataPath=benchPath+datPath+dName, 
			LabelsPath=benchPath+datPath+"Labels.csv",
			CV_RDataPath=rName,
			k=Kfold)
        # read data
        print("reading the data...")
        obj.read_data()
        # create folds if has not been done
        #if not os.path.exists(benchPath+datPath+"folds/"):
            #print("saving the folds...")
        #obj.apply_stratifiedKfold(savePath=benchPath+datPath+"folds/")
        xgb = XGBClassifier()
        lgb = LGBMClassifier()
        linsvm = LinearSVC()
        ctb = CatBoostClassifier()
        # scikit learns
        pac = PassiveAggressiveClassifier()
        sgd = SGDClassifier()
        per = Perceptron()
        knn = KNeighborsClassifier(n_neighbors=50)


        print("Linear SVM experiment")
        obj.fivefoldcv_linear_svm(path2folds= benchPath+datPath+"folds_new/", Classifier=linsvm)
        print("SGD Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=SGDClassifier(loss='log'), outFold="sgd_results_log/")
        print("SGD Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=SGDClassifier(penalty = 'l2'), outFold="sgd_results_log_pen_l2/")
        print("SGD Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=SGDClassifier(penalty = 'l1'), outFold="sgd_results_log_pen_l1/")
        print("SGD Classifier experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=SGDClassifier(penalty = 'elasticnet'), outFold="sgd_results_log_pen_elasticnet/")
        print("Perceptron experiment")
        obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=Perceptron(penalty = 'l2'), outFold="perceptron_results_log_pen_l2/")
        #print("Perceptron experiment")
        #obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=Perceptron(penalty = 'l1'), outFold="perceptron_results_log_pen_l1/")
        #print("Perceptron experiment")
        #obj.fivefoldcv_online_scikitlearn(path2folds=benchPath+datPath+"folds_new/", Classifier=Perceptron(penalty = 'elasticnet'), outFold="perceptron_results_log_pen_elasticnet/")
        print("KNN experiment")
        obj.fivefoldcv_knn(path2folds=benchPath+datPath+"folds_new/", Classifier=knn)






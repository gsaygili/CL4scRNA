import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, roc_auc_score, precision_score, recall_score
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns


class plot_save():

    def __init__(self, exp_obj, outputPath):
        self.obj = exp_obj
        self.OutputPath = outputPath
        
    def save_results(self):
        
        if not os.path.exists(self.OutputPath):
            os.makedirs(self.OutputPath)
        np.save(self.OutputPath+"/y_pred", self.obj.y_pred)
        np.save(self.OutputPath+"/truelab", self.obj.truelab)
        np.save(self.OutputPath+"/traintime", self.obj.trtime)
        np.save(self.OutputPath+"/testtime", self.obj.tetime)
        
        acc_score = accuracy_score(self.obj.truelab, self.obj.y_pred)
        bacc_score = balanced_accuracy_score(self.obj.truelab, self.obj.y_pred)
        F1_score = f1_score(self.obj.truelab, self.obj.y_pred, average="macro")
        pre_score = precision_score(self.obj.truelab, self.obj.y_pred, average='macro')
        rec_score = recall_score(self.obj.truelab, self.obj.y_pred, average='macro')
        cf_matrix = confusion_matrix(self.obj.truelab, self.obj.y_pred)
        cm = np.array2string(cf_matrix)
        with open(self.OutputPath+"/results.txt", 'w') as f:
            f.writelines("accuracy score: %.2f\n" % acc_score)
            f.writelines("balanced accuracy score: %.2f\n" % bacc_score)
            f.writelines("precision score: %.2f\n" % pre_score)
            f.writelines("recall score: %.2f\n" % rec_score)
            f.writelines("f1 score: %.2f\n" % F1_score)
            f.writelines("train time: %.2f\n" % self.obj.trtime)
            f.writelines("test time: %.2f\n\n" % self.obj.tetime)
            f.writelines("confusion matrix: \n")
            f.writelines(cm)
        
        print("==============================================================")
        print("accuracy score: %.2f" % acc_score)
        print("balanced accuracy score: %.2f" % bacc_score)
        print("precision score: %.2f" % pre_score)
        print("recall score: %.2f" % rec_score)
        print("f1 score: %.2f" % F1_score)
        print("train time: %.2f" % self.obj.trtime)
        print("test time: %.2f" % self.obj.tetime)
        print("==============================================================")
	            
    def plot_results(self):
        acc_score = accuracy_score(self.obj.truelab, self.obj.y_pred)
        bacc_score = balanced_accuracy_score(self.obj.truelab, self.obj.y_pred)
        F1_score = f1_score(self.obj.truelab, self.obj.y_pred, average="macro")
        pre_score = precision_score(self.obj.truelab, self.obj.y_pred, average='macro')
        rec_score = recall_score(self.obj.truelab, self.obj.y_pred, average='macro')
        cf_matrix = confusion_matrix(self.obj.truelab, self.obj.y_pred)
        sns.heatmap(cf_matrix, annot=True)
        print("==============================================================")
        print("accuracy score: %.2f" % acc_score)
        print("balanced accuracy score: %.2f" % bacc_score)
        print("precision score: %.2f" % pre_score)
        print("recall score: %.2f" % rec_score)
        print("f1 score: %.2f" % F1_score)
        print("train time: %.2f" % self.obj.trtime)
        print("test time: %.2f" % self.obj.tetime)
        print("==============================================================")
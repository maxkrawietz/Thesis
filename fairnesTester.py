import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix

class FairnessTester():

    def setup(self, X, sensible_att, priv_att, unpriv_att, model="prediction"):
        
        
        #set privileged confusion matrix
        priv_class = X.loc[X[sensible_att]==priv_att]["class"]
        priv_pred = X.loc[X[sensible_att]==priv_att][model]
        self.priv_confusion = confusion_matrix(priv_class, priv_pred)
        self.TP_priv = self.priv_confusion[1][1]
        self.FP_priv = self.priv_confusion[0][1]
        self.TN_priv = self.priv_confusion[0][0]
        self.FN_priv = self.priv_confusion[1][0]

        #set unprivileged confusion matrix
        unpriv_class = X.loc[X[sensible_att]==unpriv_att]["class"]
        unpriv_pred = X.loc[X[sensible_att]==unpriv_att][model]
        self.unpriv_confusion = confusion_matrix(unpriv_class, unpriv_pred)
        self.TP_unpriv = self.unpriv_confusion[1][1]
        self.FP_unpriv = self.unpriv_confusion[0][1]
        self.TN_unpriv = self.unpriv_confusion[0][0]
        self.FN_unpriv = self.unpriv_confusion[1][0]
        
        #total size
        self.priv = self.priv_confusion.sum()
        self.unpriv = self.unpriv_confusion.sum()

        # #predicted total positive/negative labels
        self.pos_pred = X.loc[X[model]==1].shape[0]
        self.neg_pred = X.loc[X[model]==0].shape[0]

        # #actual total positive/negative labels
        self.pos_act = X.loc[X[model]==1].shape[0]
        self.neg_act = X.loc[X[model]==0].shape[0]


        # #actual positive labels by group
        self.pos_act_priv = self.priv_confusion[1].sum()
        self.pos_act_unpriv = self.unpriv_confusion[1].sum()

        # #predicted positive labes by group
        self.pos_pred_priv = self.TP_priv + self.FP_priv
        self.pos_pred_unpriv = self.TP_unpriv + self.FP_unpriv

        
        # #actual negative labes by group
        self.neg_act_priv = self.priv_confusion[0].sum()
        self.neg_act_unpriv = self.unpriv_confusion[0].sum()

        # #predicted negative labels by group
        self.neg_pred_priv = self.TN_priv + self.FN_priv
        self.neg_pred_unpriv = self.TN_unpriv + self.FN_unpriv  

    def statistical_parity(self):
        priv = (self.pos_pred_priv/self.priv)
        unpriv = (self.pos_pred_unpriv/self.unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]

    def predictive_parity(self):
        priv = (self.TP_priv/self.pos_pred_priv)
        unpriv = (self.TP_unpriv/self.pos_pred_unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]
    
    def neg_predictive_parity(self):
        priv = (self.TN_priv/self.neg_pred_priv)
        unpriv = (self.TN_unpriv/self.neg_pred_unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]

    def opportunity_equality(self):
        priv = (self.TP_priv/self.pos_act_priv)
        unpriv = (self.TP_unpriv/self.pos_act_unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]

    def predictive_equality(self):
        priv = (self.FP_priv/self.neg_act_priv)
        unpriv = (self.FP_unpriv/self.neg_act_unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]
    
    def equalized_odds(self):
        connected = self.predictive_equality() + self.opportunity_equality()
        return connected

    def conditional_use_accuracy_equality(self):
        connected = self.predictive_parity() + self.neg_predictive_parity()
        return connected

    def overall_accuracy_equality(self):
        priv = (self.TP_priv+self.TN_priv)/self.priv
        unpriv = (self.TP_unpriv+self.TN_unpriv)/self.unpriv
        #ratio = unpriv/priv
        return [priv, unpriv]
    
    def treatment_equality(self):
        priv = self.predictive_equality()[0] / (self.FN_priv/self.pos_act_priv)
        unpriv = self.predictive_equality()[1] / (self.FN_unpriv/self.pos_act_unpriv)
        #ratio = unpriv/priv
        return [priv, unpriv]
    
    def confusion_based(self):
        m=[]
        m.append(self.statistical_parity())
        m.append(self.predictive_parity())
        m.append(self.neg_predictive_parity())
        m.append(self.opportunity_equality())
        m.append(self.predictive_equality())
        m.append(self.overall_accuracy_equality())
        m.append(self.treatment_equality())

        return m

    def confuison_based_dic(self):
        m ={}
        m["statistical parity"] = self.statistical_parity()
        m["predictive parity"] = self.predictive_parity()
        m["negative predictive parity"] = self.neg_predictive_parity()
        m["equal opportunity"] = self.opportunity_equality()
        m["predictive equality"] = self.predictive_equality()
        m["overall accuracy equality"] = self.overall_accuracy_equality()
        m["treatment equality"] = self.treatment_equality()

        return m
    
    def confusion_based_dic_priv(self):
    
        m ={}
        m["group"] = "priv"
        m["statistical parity"] = self.statistical_parity()[0]
        m["predictive parity"] = self.predictive_parity()[0]
        m["negative predictive parity"] = self.neg_predictive_parity()[0]
        m["equal opportunity"] = self.opportunity_equality()[0]
        m["predictive equality"] = self.predictive_equality()[0]
        m["overall accuracy equality"] = self.overall_accuracy_equality()[0]
        m["treatment equality"] = self.treatment_equality()[0]

        return m
    
    def confusion_based_dic_unpriv(self):
        m ={}
        m["group"] = "unpriv"
        m["statistical parity"] = self.statistical_parity()[1]
        m["predictive parity"] = self.predictive_parity()[1]
        m["negative predictive parity"] = self.neg_predictive_parity()[1]
        m["equal opportunity"] = self.opportunity_equality()[1]
        m["predictive equality"] = self.predictive_equality()[1]
        m["overall accuracy equality"] = self.overall_accuracy_equality()[1]
        m["treatment equality"] = self.treatment_equality()[1]

        return m

    def priv_confusion_matrix(self):
        return self.priv_confusion
    
    def unpriv_confusion_matrix(self):
        return self.unpriv_confusion
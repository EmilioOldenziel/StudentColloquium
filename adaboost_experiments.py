import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt

class AdaBoostExperiement():

    def __init__(self, depth=2, boosting_rounds=100, split_rounds=10):
        self.df_phoneme = pd.read_csv("./datasets/phoneme.dat/data", sep="\s+")
        self.df_x_phoneme = self.df_phoneme.iloc[:,0:5]
        self.df_y_phoneme = self.df_phoneme.iloc[:,5:6]

        self.n_estimators = boosting_rounds # boosting iterations
        self.max_depth = depth
        self.number_of_rounds = split_rounds # 100 for real experiment
        self.ada_err_test = np.zeros((self.number_of_rounds, self.n_estimators))
        self.ada_err_train = np.zeros((self.number_of_rounds, self.n_estimators))

    def run_experiment(self):
        for r in range(self.number_of_rounds): # number of train/test split rounds. 
            ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth),
                n_estimators=self.n_estimators,
                algorithm="SAMME.R") # init AdaBoost
            
            X_train, X_test, Y_train, Y_test = train_test_split(self.df_x_phoneme, self.df_y_phoneme, test_size=0.3) # split to train/test set (0.7, 0.3)
            
            ada.fit(X_train, np.ravel(Y_train)) # train the classifier
            
            for i, y_pred in enumerate(ada.staged_predict(X_test)):
                self.ada_err_test[r][i] = zero_one_loss(y_pred, Y_test) # validate boosting rounds with test set

            
            for i, y_pred in enumerate(ada.staged_predict(X_train)):
                self.ada_err_train[r][i] = zero_one_loss(y_pred, Y_train) # validate boosting rounds with train set

    def write_plot(self):
        fig = plt.figure()
        plt.plot(self.ada_err_train.mean(axis=0), label="Training error", color="red")
        plt.plot(self.ada_err_test.mean(axis=0), label="Test error", color="black")
        plt.ylabel("error rate")
        plt.xlabel("boosting interations")
        plt.title(f"Tree Depth: {self.max_depth}")
        plt.legend()
        #plt.show()
        fig.savefig(f"results/depth{self.max_depth}.png", dpi=fig.dpi)

ada_exp = AdaBoostExperiement(depth=8, boosting_rounds=100, split_rounds=3)
ada_exp.run_experiment()
ada_exp.write_plot()
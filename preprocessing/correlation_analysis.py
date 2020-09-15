import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
from sklearn import impute
from sklearn import preprocessing

WRITE_TO_CSV = True
MERGE_TRAIN_TEST = True

class DataCorrelation:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        # Dataframe form
        self.df_org_data = []
        # self.ls_org_data = []
        # Numpy array form
        self.ar_org_data = []

        # Name
        self.col_names = []

        # Divide the set
        self.ar_training_set = []
        self.ar_testing_set = []

        # For prediction
        self.training_data_X = []
        self.training_data_Y = []
        self.testing_data_X = []
        self.testing_data_Y = []

        # For lasso
        self.lasso_v3 = []
        self.lasso_v4 = []

    @staticmethod
    def dataframe_to_list(df):
        """Turn Dataframe into list"""
        return np.array(df).tolist()

    def read_in_data(self):
        self.df_org_data = pd.read_csv(self.input_file, sep=',', encoding='gbk')
        # encoding should be set to gbk, in this case 46 rows x 71 cols are read
        # self.ls_org_data = DataCorrelation.dataframe_to_list(self.df_org_data)
        self.ar_org_data = self.df_org_data.values
        # print(self.ar_org_data[:, 1])

    def divide_test_set(self):
        testing_set = []
        training_set = []
        for i in range(len(self.ar_org_data)):
            if i % 3 == 0:
                testing_set.append(self.ar_org_data[i, :])
            else:
                training_set.append(self.ar_org_data[i, :])
        self.ar_training_set = np.array(training_set)
        self.ar_testing_set = np.array(testing_set)
        # print(self.ar_testing_set[:, 0])

    def processing_org_data(self):
        """Specify which are the features and which are the targets"""
        # Also in the column['Gender'], we use 1 to represent 男, 0 to represent 女
        training_data_Y = self.ar_training_set[:, 2:4]
        training_data_X = self.ar_training_set[:, 4:]
        testing_data_Y = self.ar_testing_set[:, 2:4]
        testing_data_X = self.ar_testing_set[:, 4:]
        # print(training_data_X[:, 3])
        for i in range(len(training_data_X)):
            if training_data_X[i][3] == "男":
                training_data_X[i][3] = 1
            else:
                training_data_X[i][3] = 0
        for j in range(len(testing_data_X)):
            if testing_data_X[j][3] == "男":
                testing_data_X[j][3] = 1
            else:
                testing_data_X[j][3] = 0
        # print(training_data_X[:, 3])
        self.testing_data_X = testing_data_X
        self.training_data_X = training_data_X
        self.testing_data_Y = testing_data_Y
        self.training_data_Y = training_data_Y

    def data_imputation(self):
        """Imputate data with Means"""

        # KNN Imputer
        imp1 = impute.KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        self.training_data_X = imp1.fit_transform(self.training_data_X)
        imp2 = impute.KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        self.testing_data_X = imp2.fit_transform(self.testing_data_X)
        # np.savetxt("C:/Users/lihanmin/Desktop/data_processing/temp1.csv", self.training_data_X, delimiter=',')

        # Simple Imputer with 'mean' strategy
        # imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        # self.training_data_X = imp.fit_transform(self.training_data_X)
        # np.savetxt("C:/Users/lihanmin/Desktop/data_processing/temp2.csv", self.training_data_X, delimiter=',')

        if WRITE_TO_CSV:
            # Write Into CSV
            training_set_temp = np.concatenate((self.training_data_Y, self.training_data_X), axis=1)
            training_set = np.concatenate((self.ar_training_set[:, 0:2], training_set_temp), axis=1)
            test_set_temp = np.concatenate((self.testing_data_Y, self.testing_data_X), axis=1)
            test_set = np.concatenate((self.ar_testing_set[:, 0:2], test_set_temp), axis=1)
            train_num = len(training_set)
            test_num = len(test_set)
            # print(train_num, test_num)
            for i in range(train_num):
                if training_set[i][7] == 0:
                    training_set[i][7] = '女'
                else:
                    training_set[i][7] = '男'
            for j in range(test_num):
                if test_set[j][7] == 0:
                    test_set[j][7] = '女'
                else:
                    test_set[j][7] = '男'
            # print(pd.__version__)
            # print(type(self.df_org_data.columns))

            # Get names
            col_names = []
            for item in self.df_org_data.columns:
                col_names.append(item)

            # Merge
            if MERGE_TRAIN_TEST:
                total_length = len(self.ar_org_data)
                total_data = []
                idx_train = 0
                idx_test = 0
                for i in range(total_length):
                    if i % 3 == 0:
                        total_data.append(test_set[idx_test])
                        idx_test += 1
                    else:
                        total_data.append(training_set[idx_train])
                        idx_train += 1
                total_data = pd.DataFrame(total_data)
                total_data.columns = col_names
                total_file = 'C:/Users/lihanmin/Desktop/data_processing/Imputed_data.csv'
                total_data.to_csv(total_file, sep=',', encoding='gbk', index=False)
                exit()

            # Whole data
            test_set = pd.DataFrame(test_set)
            training_set = pd.DataFrame(training_set)

            # Set name
            test_set.columns = col_names
            training_set.columns = col_names

            # Write
            train_file = 'C:/Users/lihanmin/Desktop/data_processing/train.csv'
            test_file = 'C:/Users/lihanmin/Desktop/data_processing/test.csv'
            test_set.to_csv(test_file, sep=',', encoding='gbk', index=False)
            training_set.to_csv(train_file, sep=',', encoding='gbk', index=False)
            exit()

        # Normalization
        self.training_data_X = preprocessing.scale(self.training_data_X)
        self.testing_data_X = preprocessing.scale(self.testing_data_X)
        # np.savetxt("C:/Users/lihanmin/Desktop/data_processing/temp3.csv", self.training_data_X, delimiter=',')

    def lasso_regression(self):
        alphas = [0.0005, 0.001, 0.1, 1, 3]
        # To find the best alpha
        model_lasso = linear_model.MultiTaskLassoCV(alphas=alphas, max_iter=100000)
        model_lasso.fit(self.training_data_X, self.training_data_Y)
        # print(model_lasso.coef_)
        coef_for_v3, coef_for_v4 = model_lasso.coef_

        coef_num = len(coef_for_v3)
        for i in range(coef_num):
            if coef_for_v3[i] != 0:
                self.lasso_v3.append(i + 5)
            if coef_for_v4[i] != 0:
                self.lasso_v4.append(i + 5)
        return self.lasso_v3, self.lasso_v4


def main():
    # Display Issue
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    np.set_printoptions(threshold=np.inf)

    input_file = 'C:/Users/lihanmin/Desktop/data_processing/raw_data.CSV'
    output_file = 'C:/Users/lihanmin/Desktop/data_processing/result.CSV'
    dataset = DataCorrelation(input_file, output_file)
    dataset.read_in_data()
    dataset.divide_test_set()
    dataset.processing_org_data()
    dataset.data_imputation()

    # LASSO prediction
    tgt1, tgt2 = dataset.lasso_regression()
    print("Features for 呼吸困难指标: ")
    print(tgt1, end="  ")
    print("\n--------------------------------")
    print("Features for SGRQ: ")
    print(tgt2, end="  ")
    return


if __name__ == '__main__':
    main()

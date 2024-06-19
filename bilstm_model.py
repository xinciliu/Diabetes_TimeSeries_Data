import os, math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, trange
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization, Bidirectional
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
from xgboost import XGBClassifier
import ast
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#两种方式：将one-hot encoding拆解成多维度输入 / 保持one-hot encoding 单维度输入
#方法1
def change_dict(lis):
    dic = {}
    for i in range(len(lis)):
        dic[i] = lis[i]
    return dic

def re_originze_feature_file(filename):
    """
    拆解encoding features
    """
    feature_df = pd.read_csv(filename, sep="\t", index_col=0)
    #把one-hot encoding变成多列
    #Diabetic_Macrovascular_Complications
    Diabetic_Macrovascular_Complications = ['cerebrovascular disease', 'coronary heart disease', 'peripheral arterial disease']
    feature_df[Diabetic_Macrovascular_Complications]=feature_df['Diabetic_Macrovascular_Complications'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(pd.Series)
    feature_df=feature_df.drop("Diabetic_Macrovascular_Complications", axis = 1)
    #Diabetic_Microvascular_Complications
    Diabetic_Microvascular_Complications = ['nephropathy', 'retinopathy', 'neuropathy']
    feature_df[Diabetic_Microvascular_Complications]=feature_df['Diabetic_Microvascular_Complications'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(pd.Series)
    feature_df=feature_df.drop("Diabetic_Microvascular_Complications", axis = 1)
    #Comorbidities
    Comorbidities_value = ['hypoleukocytemia', 'pulmonary nodule', 'cataract', 'hepatic dysfunction', 'hypokalemia', 'urinary tract infection', 'osteoporosis', 'lumbar spine tumor', 'hypertension', 'cholelithiasis', 'chronic gastritis', 'fatty liver disese', 'hypocalcemia', 'prostatic hyperplasia', 'sinus bradycardia', 'leucopenia', 'chronic hepatitis B', 'thyroid nodule', 'nephrolithiasis', 'systemic sclerosis', 'hyperlipidemia', 'sinus arrhythmia', 'vitamin D deficiency', 'breast cancer', 'fatty liver disease']
    feature_df[Comorbidities_value]=feature_df['Comorbidities_value'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(pd.Series)
    feature_df=feature_df.drop("Comorbidities_value", axis = 1)
    #Hypoglycemic_Agents
    Hypoglycemic_Agents = ['acarbose', 'liraglutide', 'Humulin R', 'metformin', 'insulin detemir', 'Novolin R', 'Novolin 50R', 'insulin aspart', 'insulin aspart 70/30', 'insulin degludec', 'dapagliflozin', 'voglibose', 'insulin glargine']
    feature_df[Hypoglycemic_Agents]=feature_df['Hypoglycemic_Agents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(pd.Series)
    feature_df=feature_df.drop("Hypoglycemic_Agents", axis = 1)
    #Other_Agents
    Other_Agents = ['calcium carbonate and vitamin D3 tablet', 'benidipine', 'calcitriol', 'epalrestat', 'valsartan', 'metoprolol', 'calcium carbonate', 'raberazole', 'olmesartan medoxomil', 'beiprostaglandin sodium', 'calcium dobesilate', 'trimetazidine', 'diammonium glycyrrhizinate', 'aspirin', 'amlodipine', 'isosorbide mononitrate', 'vitamin B1', 'mecobalamin', 'rosuvastatin', 'leucogen', 'clopidogrel', 'atorvastatin', 'potassium chloride', 'losartan']
    feature_df[Other_Agents]=feature_df['Other_Agents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(pd.Series)
    feature_df= feature_df.drop("Other_Agents", axis = 1)
    return feature_df

def remove_onehot_encoding_columns(filename):
    """
    删除one-hot encodings
    """
    feature_df = pd.read_csv(filename, sep="\t", index_col=0)
    #把one-hot encoding删除
    feature_df=feature_df.drop("Diabetic_Macrovascular_Complications", axis = 1)
    feature_df=feature_df.drop("Diabetic_Microvascular_Complications", axis = 1)
    feature_df=feature_df.drop("Comorbidities_value", axis = 1)
    feature_df=feature_df.drop("Hypoglycemic_Agents", axis = 1)
    feature_df= feature_df.drop("Other_Agents", axis = 1)
    return feature_df

def filling_nan(df):
    """
    把dataframe中所有nan的情况用全列的均值fill in
    """
    columns = df.columns
    for col in columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def running_bilstm(filename):
    """
    训练bilstm模型 - 得到预测的结论
    """
    feature_df = remove_onehot_encoding_columns(filename)
    #神经网络的训练：避免feature_df中出现np.nan    
    feature_df = filling_nan(feature_df)
    print(feature_df)
    X = feature_df.drop("CGM (mg / dl)", axis = 1)
    y = feature_df["CGM (mg / dl)"]
    X_training = X[:int(0.8*len(X))]
    X_testing = X[int(0.8*len(X)):]
    y_training = y[:int(0.8*len(y))]
    y_testing = y[int(0.8*len(y)):]
    y_training = le.fit_transform(y_training)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(X_training)
    scaler_y.fit(y_training.reshape(-1, 1))
    scaled_X_train_data = scaler_x.transform(X_training)
    scaled_y_train_data = scaler_y.transform(y_training.reshape(-1, 1))
    scaled_X_train_data = np.reshape(scaled_X_train_data, (scaled_X_train_data.shape[0], 1, scaled_X_train_data.shape[1]))
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape = (scaled_X_train_data.shape[1], scaled_X_train_data.shape[2]))))
    model.add(Dense(150, activation = 'relu'))
    model.add(Dropout(0.20))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, activation = 'relu'))
    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit(scaled_X_train_data, scaled_y_train_data, epochs = 200, batch_size = 32, shuffle = False)
    scaled_X_test_data = scaler_x.transform(X_testing)
    scaled_X_test_data = np.reshape(scaled_X_test_data, (scaled_X_test_data.shape[0], 1, scaled_X_test_data.shape[1]))
    prediction = model.predict(scaled_X_test_data, batch_size = 32)
    scaled_prediction = scaler_y.inverse_transform(prediction)
    x = scaled_prediction.reshape(1, len(scaled_prediction))[0]
    return x, y_testing


if __name__ == "__main__":
    feature_filename = sys.argv[1]
    running_bilstm(feature_filename)

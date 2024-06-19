# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
XGB 方法分析提取出来的feature data
"""
import pandas as pd
from xgboost import XGBClassifier
import ast
import sys

def change_dict(lis):
    dic = {}
    for i in range(len(lis)):
        dic[i] = lis[i]
    return dic

def re_originze_feature_file(filename):
    """
    按照xgb的要求重新整理一下feature，one-encoding从列表分为多列
    """
    feature_df = pd.read_csv(filename, sep="\t", index_col=0)
    #由于使用xgb，要把one-hot encoding变成多列
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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def running_xgb(filename):
    """
    训练xgb模型 - 得到预测的结论
    """
    feature_df = re_originze_feature_file(filename)
    X = feature_df.drop("CGM (mg / dl)", axis = 1)
    y = feature_df["CGM (mg / dl)"]
    X_training = X[:int(0.8*len(X))]
    X_testing = X[int(0.8*len(X)):]
    y_training = y[:int(0.8*len(y))]
    y_testing = y[int(0.8*len(y)):]
    y_training = le.fit_transform(y_training)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = None, random_state=1)
    model = XGBClassifier()
    model.fit(X_training, y_training)
    d = model.predict(X_testing)
    return d, y_testing

if __name__ == "__main__":
    feature_filename = sys.argv[1]
    running_xgb(feature_filename)

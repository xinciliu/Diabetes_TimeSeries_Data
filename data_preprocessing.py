import pandas as pd
import os, math
import numpy as np
#read the summary data
def read_summary_data(filepath):
    df = pd.read_excel(filepath)
    return df
def read_each_day(TDM, Patient_Number):
    try:
        df = pd.read_excel("./" + TDM + "/" + Patient_Number + ".xls")
    except:
        df = pd.read_excel("./" + TDM + "/" + Patient_Number + ".xlsx")
    return df
df_summary=read_summary_data("Shanghai_T1DM_Summary.xlsx")
def transdering_onehot_encoder(onehot_list_dict, x):
    """
    对某一列进行one-hot encoding
    """
    n = len(onehot_list_dict)
    one_hot_list = [0]*n
    for i in range(len(onehot_list_dict)):
        word = onehot_list_dict[i]
        if word in x:
            one_hot_list[i] = 1
    return one_hot_list
def encoding_summary_total(filepath):
    """
    把summary部分的数据进行处理
    """
    df_summary=read_summary_data(filepath)
    #drinker encoding
    df_summary.loc[df_summary["Alcohol Drinking History (drinker/non-drinker)"] == "drinker", "Alcohol Drinking History (drinker/non-drinker)"] = "1"
    df_summary.loc[df_summary["Alcohol Drinking History (drinker/non-drinker)"] == "non-drinker", "Alcohol Drinking History (drinker/non-drinker)"] = "0"
    #Type of Diabetes
    df_summary.loc[df_summary["Type of Diabetes"] == "T1DM", "Type of Diabetes"] = "1"
    df_summary.loc[df_summary["Type of Diabetes"] == "T2DM", "Type of Diabetes"] = "2"
    #Acute Diabetic Complications
    df_summary.loc[df_summary["Acute Diabetic Complications"] == "none", "Acute Diabetic Complications"] = "0"
    df_summary.loc[df_summary["Acute Diabetic Complications"] == "diabetic ketoacidosis", "Acute Diabetic Complications"] = "1"
    #Diabetic Macrovascular  Complications
    Diabetic_Macrovascular_Complications = ['cerebrovascular disease', 'coronary heart disease', 'peripheral arterial disease']
    df_summary["Diabetic_Macrovascular_Complications"] = df_summary["Diabetic Macrovascular  Complications"].apply(lambda x: transdering_onehot_encoder(Diabetic_Macrovascular_Complications, x))
    df_summary = df_summary.drop("Diabetic Macrovascular  Complications", axis=1)
    #Diabetic Microvascular Complications
    Diabetic_Microvascular_Complications = ['nephropathy', 'retinopathy', 'neuropathy']
    df_summary["Diabetic_Microvascular_Complications"] = df_summary["Diabetic Microvascular Complications"].apply(lambda x: transdering_onehot_encoder(Diabetic_Microvascular_Complications, x))
    df_summary = df_summary.drop("Diabetic Microvascular Complications", axis=1)
    #Comorbidities
    Comorbidities_value = ['hypoleukocytemia', 'pulmonary nodule', 'cataract', 'hepatic dysfunction', 'hypokalemia', 'urinary tract infection', 'osteoporosis', 'lumbar spine tumor', 'hypertension', 'cholelithiasis', 'chronic gastritis', 'fatty liver disese', 'hypocalcemia', 'prostatic hyperplasia', 'sinus bradycardia', 'leucopenia', 'chronic hepatitis B', 'thyroid nodule', 'nephrolithiasis', 'systemic sclerosis', 'hyperlipidemia', 'sinus arrhythmia', 'vitamin D deficiency', 'breast cancer', 'fatty liver disease']
    df_summary["Comorbidities_value"] = df_summary["Comorbidities"].apply(lambda x: transdering_onehot_encoder(Comorbidities_value, x))
    df_summary = df_summary.drop("Comorbidities", axis=1)
    #Hypoglycemic Agents
    Hypoglycemic_Agents = ['acarbose', 'liraglutide', 'Humulin R', 'metformin', 'insulin detemir', 'Novolin R', 'Novolin 50R', 'insulin aspart', 'insulin aspart 70/30', 'insulin degludec', 'dapagliflozin', 'voglibose', 'insulin glargine']
    df_summary["Hypoglycemic_Agents"] = df_summary["Hypoglycemic Agents"].apply(lambda x: transdering_onehot_encoder(Hypoglycemic_Agents, x))
    df_summary = df_summary.drop("Hypoglycemic Agents", axis=1)
    #Other Agents
    Other_Agents = ['calcium carbonate and vitamin D3 tablet', 'benidipine', 'calcitriol', 'epalrestat', 'valsartan', 'metoprolol', 'calcium carbonate', 'raberazole', 'olmesartan medoxomil', 'beiprostaglandin sodium', 'calcium dobesilate', 'trimetazidine', 'diammonium glycyrrhizinate', 'aspirin', 'amlodipine', 'isosorbide mononitrate', 'vitamin B1', 'mecobalamin', 'rosuvastatin', 'leucogen', 'clopidogrel', 'atorvastatin', 'potassium chloride', 'losartan']
    df_summary["Other_Agents"] = df_summary["Other Agents"].apply(lambda x: transdering_onehot_encoder(Other_Agents, x))
    df_summary = df_summary.drop("Other Agents", axis=1)
    #Hypoglycemia (yes/no)
    df_summary.loc[df_summary["Hypoglycemia (yes/no)"] == "yes", "Hypoglycemia (yes/no)"] = "1"
    df_summary.loc[df_summary["Hypoglycemia (yes/no)"] == "no", "Hypoglycemia (yes/no)"] = "0"
    return df_summary
import re
df = read_each_day("Shanghai_T1DM", "1002_0_20210504")
def encode_date(x):
    """
    encode 时间点 - 取当时时间的小时-分钟作为时间节点，例如1033，1048等
    """
    x = str(x)
    return "".join(x.split(" ")[1].split(":")[:2])

def encode_dietary_Insulin(x):
    """
    encode 摄入的食物或者Insulin
    """
    if (not x) or (x == "data not available"):
        return 0
    else:
        numbers = re.findall(r'\d+', str(x))
        number = 0
        for val in numbers:
            number += int(val)
    return number

def CSII_basal_insulin(data_lis):
    """
    encoding CSII_basal_insulin
    """
    for i in range(len(data_lis)):
        if str(data_lis[i]) == "nan":
            data_lis[i] = data_lis[i-1]
    for i in range(len(data_lis)):
        if data_lis[i] == "temporarily suspend insulin delivery":
            data_lis[i] = 0
    return data_lis

def encoding_each_day_timeseries(filepath, patientid):
    """
    """
    df = read_each_day(filepath, patientid)
    #Date
    df["date"] = df["Date"].apply(lambda x: encode_date(x))
    df = df.drop("Date", axis = 1)
    df = df.drop("CBG (mg / dl)", axis = 1)
    df = df.drop("Blood Ketone (mmol / L)", axis = 1)
    df = df.drop("饮食", axis = 1)
    df = df.drop("Insulin dose - i.v.", axis = 1)
    #Dietary intake	
    df["Dietary_intake"] = df["Dietary intake"].apply(lambda x: encode_dietary_Insulin(x))
    df = df.drop("Dietary intake", axis = 1)
    #Insulin dose - s.c.
    df["Insulin_dose_sc"] = df["Insulin dose - s.c."].apply(lambda x:encode_dietary_Insulin(x))
    df["non_insulin_hypoglycemic_agents"] = df["Non-insulin hypoglycemic agents"].apply(lambda x:encode_dietary_Insulin(x))
    df["CSII_bolus_insulin"] = df["CSII - bolus insulin (Novolin R, IU)"].apply(lambda x:encode_dietary_Insulin(x))
    df = df.drop("Insulin dose - s.c.", axis = 1)
    df = df.drop("Non-insulin hypoglycemic agents", axis = 1)
    df = df.drop("CSII - bolus insulin (Novolin R, IU)", axis = 1)
    #encoding CSII - basal insulin (Novolin R, IU / H)
    data_lis = list(df["CSII - basal insulin (Novolin R, IU / H)"])
    data_lis = CSII_basal_insulin(data_lis)
    df["CSII_basal_insulin"] = data_lis
    df = df.drop("CSII - basal insulin (Novolin R, IU / H)", axis = 1)
    return df


#生成整体的feature table
def summarize_datasets_together(DM_type, patient_id):
    """
    summarize datasets数据
    """
    time_series_df = encoding_each_day_timeseries(DM_type, patient_id)
    summary_df = encoding_summary_total(DM_type + "_Summary.xlsx")
    summary_part = summary_df.loc[summary_df['Patient Number'] == patient_id]
    summary_part = summary_part.drop("Patient Number", axis = 1)
    dc = pd.concat([summary_part] * len(time_series_df), ignore_index=True)
    all_feature_table = pd.concat([time_series_df, dc], axis = 1)
    return all_feature_table

#以其中一个patient_id作 training - testing dataset,跑出他的feature data
feature_df = summarize_datasets_together("Shanghai_T1DM", "1002_0_20210504")
feature_df.to_csv("feature_data.txt", sep="\t")

#以所有patient_id做为 training - testing dataset, 跑出feature data
#Shanghai_T1DM
pd.set_option("future.no_silent_downcasting", True)
T1DM_patient_id = ["1001_0_20210730", "1002_0_20210504", "1002_1_20210521", "1002_2_20210909", "1003_0_20210831","1004_0_20210425", \
"1005_0_20210522","1006_0_20210114", "1006_1_20210209", "1006_2_20210303", "1007_0_20210726", "1008_0_20210713", "1009_0_20210803", \
"1010_0_20210915", "1011_0_20210622", "1012_0_20210923"]
T1DM_patient_id = [
    "1002_0_20210504", "1002_1_20210521","1005_0_20210522", "1006_0_20210114", "1006_1_20210209","1006_2_20210303","1007_0_20210726",\
    "1010_0_20210915", "1011_0_20210622"]
feature_df_list = []
for patient_id in T1DM_patient_id:
    feature_df = summarize_datasets_together("Shanghai_T1DM", patient_id)
    feature_df = feature_df.replace("/", np.nan)
    feature_df_list.append(feature_df)
vertical_concat = pd.concat(feature_df_list, axis=0)
vertical_concat.to_csv("all_feature_data.txt", sep="\t")

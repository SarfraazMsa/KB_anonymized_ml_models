#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 01:02:38 2022

@author: root
"""


import pandas as pd #Importing Pandas which is a data analysis package of python
import numpy as np #Importing Numpy which is numerical package of python
from pandas import ExcelWriter #Importing ExcelWriter to save the necessary outcomes in Excel Format
from sqlalchemy import create_engine
import psycopg2
import os  #Import operating System
import os.path #os.path to read the file from the desired location
import datetime
from datetime import timedelta
    
#import matplotlib.pyplot as plt
import time
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
import numpy as np
# to measure exec time
from timeit import default_timer as timer
import random
import functools
random.seed(3)
import sqlalchemy
from sqlalchemy import create_engine
import statsmodels.api as sm

import statsmodels.formula.api as smf

import datetime
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
''' Deep Neural network with 5 layers '''
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
tf.random.set_seed(89)
import random as rn
rn.seed(1254)
''' just need to change the location and observe the plots in the python environment (spyder)'''


os.chdir("/home/sarfraaz/Videos/kb_smu_child_malnutrition")

''' the calculation are already explained in the script - smu_severe_underweight_model.py'''

''' all though the logic of the outcome will be explained in line 53 and 66'''

df = os.path.join("Master_children_dataset_Udaipur.csv")
df = pd.read_csv(df)
zz=['Treatment still in progress',"Don't know about the final status",'The child is cured by taking care of eating at home','Treatment complete at PHC/CHC','Death',"Parents Don't want to go to Hospital",'Treatment complete at DH','Treatment Complete at MTC','Child is underweight' ]

z1=df[df['treatment_outcome_by_kb_monitors'].isin(zz)]

''' sam_confirmed is the outcome variable - if values from treatment_outcome_by_kb_monitors is in zz list then 1 else 0'''

df['sam_confirmed']=np.where((df['treatment_outcome_by_kb_monitors'].isin(zz)),1,0)



df['child_linked_to_mother']=np.where((df['pregnancy_id'].notnull()),1,0)
df['first_date_of_hh_visit_by_kb_monitors']= pd.to_datetime(df['first_date_of_hh_visit_by_kb_monitors'], errors='coerce').dt.date
df['second_date_of_hh_visit_by_kb_monitors']= pd.to_datetime(df['second_date_of_hh_visit_by_kb_monitors'], errors='coerce').dt.date
df['third_date_of_hh_visit_by_kb_monitors']= pd.to_datetime(df['third_date_of_hh_visit_by_kb_monitors'], errors='coerce').dt.date
df['first_date_of_hh_visit_by_kb_monitors']=df['first_date_of_hh_visit_by_kb_monitors'].fillna(df['third_date_of_hh_visit_by_kb_monitors'])
df['first_date_of_hh_visit_by_kb_monitors']=df['first_date_of_hh_visit_by_kb_monitors'].fillna(df['second_date_of_hh_visit_by_kb_monitors'])

''' Just select those children where monitors have made the household visit'''
df=df[df['household_visit_by_kb_monitors']=="Yes"]


df['anemia_1']=np.where((df['hb_1']<=8),1,0)
df['anemia_2']=np.where((df['hb_2']<=8),1,0)
df['anemia_3']=np.where((df['hb_3']<=8),1,0)
df['anemia_4']=np.where((df['hb_4']<=8),1,0)

df['anc_date_1'] = pd.to_datetime(df['anc_date_1'], errors='coerce').dt.date
df['anc_date_2'] = pd.to_datetime(df['anc_date_2'], errors='coerce').dt.date
df['anc_date_3'] = pd.to_datetime(df['anc_date_3'], errors='coerce').dt.date
df['anc_date_4'] = pd.to_datetime(df['anc_date_4'], errors='coerce').dt.date
df['lmp_date'] = pd.to_datetime(df['lmp_date'], errors='coerce').dt.date

df['lmp_date'] = pd.to_datetime(df['lmp_date'], errors='coerce').dt.date
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce').dt.date
df['vaccine_date_opv_3'] = pd.to_datetime(df['vaccine_date_opv_3'], errors='coerce').dt.date
df['vaccine_date_penta_3'] = pd.to_datetime(df['vaccine_date_penta_3'], errors='coerce').dt.date
df['checkup_date_1'] = pd.to_datetime(df['checkup_date_1'], errors='coerce').dt.date
df['checkup_date_2'] = pd.to_datetime(df['checkup_date_2'], errors='coerce').dt.date
df['checkup_date_3'] = pd.to_datetime(df['checkup_date_3'], errors='coerce').dt.date
df['checkup_date_4'] = pd.to_datetime(df['checkup_date_4'], errors='coerce').dt.date
df['checkup_date_5'] = pd.to_datetime(df['checkup_date_5'], errors='coerce').dt.date

df['first_diff'] = (df['anc_date_1']-df['lmp_date']).dt.days
df['second_diff'] = (df['anc_date_2']-df['lmp_date']).dt.days
df['third_diff'] = (df['anc_date_3']-df['lmp_date']).dt.days
df['fourth_diff'] = (df['anc_date_4']-df['lmp_date']).dt.days    

df['penta3_dob'] = (df['vaccine_date_penta_3']-df['date_of_birth']).dt.days 
df['opv3_dob'] = (df['vaccine_date_opv_3']-df['date_of_birth']).dt.days 
df['min_dose3'] = df[['penta3_dob','opv3_dob']].max(axis=1)

df['timely_dose3_vaccination_within_6_months']=np.where(((df['min_dose3']<=180) & (df['vaccine_status_penta_3']=="Given")) | ((df['min_dose3']<=180) & (df['vaccine_status_opv_3']=="Given")),1,0)
df['first_diffc'] = (df['checkup_date_1']-df['date_of_birth']).dt.days
df['second_diffc'] = (df['checkup_date_2']-df['date_of_birth']).dt.days
df['third_diffc'] = (df['checkup_date_3']-df['date_of_birth']).dt.days
df['fourth_diffc'] = (df['checkup_date_4']-df['date_of_birth']).dt.days    
df['fifth_diffc'] = (df['checkup_date_5']-df['date_of_birth']).dt.days


df['anemia_1_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']<=90)) | ((df['hb_2']<=8) & (df['first_diff']<=90)) | ((df['hb_3']<=8) & (df['first_diff']<=90)) | ((df['hb_4']<=8) & (df['first_diff']<=90)) | ((df['hb_1']<=8) & (df['second_diff']<=90)) | ((df['hb_2']<=8) & (df['second_diff']<=90)) | ((df['hb_3']<=8) & (df['second_diff']<=90)) | ((df['hb_4']<=8) & (df['second_diff']<=90)) | ((df['hb_1']<=8) & (df['third_diff']<=90)) | ((df['hb_2']<=8) & (df['third_diff']<=90)) | ((df['hb_3']<=8) & (df['third_diff']<=90)) | ((df['hb_4']<=8) & (df['third_diff']<=90)),1,0)
df['anemia_2_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_2']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_3']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_4']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_1']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_2']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_3']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_4']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_1']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_2']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_3']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_4']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)),1,0)
df['anemia_3_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_2']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_3']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_4']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_1']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_2']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_3']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_4']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_1']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_2']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_3']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_4']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)),1,0)

df['hypertension_1_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']<=90)),1,0)
df['hypertension_2_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)),1,0)
df['hypertension_3_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)),1,0)

df['weight_2_months']=np.where(((df['first_diffc']>=45) & (df['first_diffc']<75)),df['weight_kg_1'],np.nan)  
df['weight_2_months']=np.where(((df['second_diffc']>=45) & (df['second_diffc']<75)),df['weight_kg_2'],df['weight_2_months']) 
df['weight_2_months']=np.where(((df['third_diffc']>=45) & (df['third_diffc']<75)),df['weight_kg_3'],df['weight_2_months']) 
df['weight_2_months']=np.where(((df['fourth_diffc']>=45) & (df['fourth_diffc']<75)),df['weight_kg_4'],df['weight_2_months']) 
df['weight_2_months']=np.where(((df['fifth_diffc']>=45) & (df['fifth_diffc']<75)),df['weight_kg_5'],df['weight_2_months']) 

df['weight_4_months']=np.where(((df['first_diffc']>=105) & (df['first_diffc']<135)),df['weight_kg_1'],np.nan)  
df['weight_4_months']=np.where(((df['second_diffc']>=105) & (df['second_diffc']<135)),df['weight_kg_2'],df['weight_4_months']) 
df['weight_4_months']=np.where(((df['third_diffc']>=105) & (df['third_diffc']<135)),df['weight_kg_3'],df['weight_4_months']) 
df['weight_4_months']=np.where(((df['fourth_diffc']>=105) & (df['fourth_diffc']<135)),df['weight_kg_4'],df['weight_4_months']) 
df['weight_4_months']=np.where(((df['fifth_diffc']>=105) & (df['fifth_diffc']<135)),df['weight_kg_5'],df['weight_4_months']) 

df['weight_6_months']=np.where(((df['first_diffc']>=165) & (df['first_diffc']<195)),df['weight_kg_1'],np.nan)  
df['weight_6_months']=np.where(((df['second_diffc']>=165) & (df['second_diffc']<195)),df['weight_kg_2'],df['weight_6_months']) 
df['weight_6_months']=np.where(((df['third_diffc']>=165) & (df['third_diffc']<195)),df['weight_kg_3'],df['weight_6_months']) 
df['weight_6_months']=np.where(((df['fourth_diffc']>=165) & (df['fourth_diffc']<195)),df['weight_kg_4'],df['weight_6_months']) 
df['weight_6_months']=np.where(((df['fifth_diffc']>=165) & (df['fifth_diffc']<195)),df['weight_kg_5'],df['weight_6_months']) 

  
df['hypertension_1']=np.where((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140),1,0)
df['hypertension_2']=np.where((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140),1,0)
df['hypertension_3']=np.where((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140),1,0)
df['hypertension_4']=np.where((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140),1,0)

df['shypertension_1']=np.where((df['bp_dia_1']>=110) | (df['bp_sys_1']>=160),1,0)
df['shypertension_2']=np.where((df['bp_dia_2']>=110) | (df['bp_sys_2']>=160),1,0)
df['shypertension_3']=np.where((df['bp_dia_3']>=110) | (df['bp_sys_3']>=160),1,0)
df['shypertension_4']=np.where((df['bp_dia_4']>=110) | (df['bp_sys_4']>=160),1,0)

df['high_risk_1']=np.where((df['referral_facility_name_hr_1_mother'].notnull()) ,1,0)
df['high_risk_2']=np.where((df['referral_facility_name_hr_2_mother'].notnull()) ,1,0)
df['high_risk_3']=np.where((df['referral_facility_name_hr_3_mother'].notnull()) ,1,0)
df['high_risk_4']=np.where((df['referral_facility_name_hr_4_mother'].notnull()) ,1,0)


df['high_risk_mother']=np.where((df['high_risk_1']>0) | (df['high_risk_2']>0) | (df['high_risk_3']>0) | (df['high_risk_4']>0) ,1,0)



df['low_birth_weight']=np.where((df['birth_weight_kg']<2.5) ,1,0)


cols=['weight_2_months','weight_4_months','weight_6_months']
for col in cols:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    


df['mother_age_less_than_18']=np.where((df['mother_age']<18),1,0)  
df['mother_age_greater_than_35']=np.where((df['mother_age']>=35),1,0) 
df['pregnancy_greater_than_3']=np.where((df['pregnancy_no']>3),1,0) 
df['tt_booster_given']=np.where((df['tt_booster_date'].notnull()),1,0) 

df['low_height']=np.where((df['height']<140),1,0) 
df['mother_age_greater_than_20_less_than_25']=np.where((df['mother_age']>=20) & (df['mother_age']<25),1,0) 
df['mother_age_greater_than_25_less_than_30']=np.where((df['mother_age']>=25) & (df['mother_age']<30),1,0) 
df['mother_age_greater_than_30_less_than_35']=np.where((df['mother_age']>=30) & (df['mother_age']<35),1,0) 
df['mother_age_greater_than_20_less_than_30']=np.where((df['mother_age']>=20) & (df['mother_age']<30),1,0) 

child_sam = os.path.join("Combo sheet - malnutrition(1).xlsx")  # importing the file

child_sam = pd.read_excel(child_sam, sheet_name='Month Wise Data')
child_sam['number_of_children_under_5_registered_per_kb_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_children_under_5_registered_per_kb_rch_dataset'].transform('sum')
child_sam['number_of_pregnant_women_registered_per_kb_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_pregnant_women_registered_per_kb_rch_dataset'].transform('sum')
child_sam['number_of_suspected_sam_household_visits_by_kb_monitor']=child_sam.groupby(['anganwadi_id'])['number_of_suspected_sam_household_visits_by_kb_monitor'].transform('sum')
child_sam['number_of_sam_children_identified_by_kb_monitor']=child_sam.groupby(['anganwadi_id'])['number_of_sam_children_identified_by_kb_monitor'].transform('sum')


child_sam['number_of_pregnant_women_with_hb_<=_8_from_kb_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_pregnant_women_with_hb_<=_8_from_kb_rch_dataset'].transform('sum')

child_sam['number_of_severe_underweight_for_age_children_from_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_severe_underweight_for_age_children_from_rch_dataset'].transform('sum')
child_sam['number_of_pregnant_women_with_hypertension_from_kb_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_pregnant_women_with_hypertension_from_kb_rch_dataset'].transform('sum')
child_sam['number_of_pregnant_women_with_diabetes_from_kb_rch_dataset']=child_sam.groupby(['anganwadi_id'])['number_of_pregnant_women_with_diabetes_from_kb_rch_dataset'].transform('sum')


child_sam['proportion_of_anemic_cases_reported_in_village']=child_sam['number_of_pregnant_women_with_hb_<=_8_from_kb_rch_dataset']*100/child_sam['number_of_pregnant_women_registered_per_kb_rch_dataset']
child_sam['proportion_of_hypertension_cases_reported_in_village']=child_sam['number_of_pregnant_women_with_hypertension_from_kb_rch_dataset']/child_sam['number_of_pregnant_women_registered_per_kb_rch_dataset']
child_sam['proportion_of_diabetes_cases_reported_in_village']=child_sam['number_of_pregnant_women_with_diabetes_from_kb_rch_dataset']/child_sam['number_of_pregnant_women_registered_per_kb_rch_dataset']
child_sam['proportion_of_suspected_sam_cases_reported_in_village']=child_sam['number_of_severe_underweight_for_age_children_from_rch_dataset']/child_sam['number_of_children_under_5_registered_per_kb_rch_dataset']
child_sam['proportion_of_confirmed_sam_cases_reported_in_village']=child_sam['number_of_sam_children_identified_by_kb_monitor']*100/child_sam['number_of_suspected_sam_household_visits_by_kb_monitor']


child_sam=(child_sam.drop_duplicates(subset='anganwadi_id', keep='first'))[['anganwadi_id','proportion_of_confirmed_sam_cases_reported_in_village','proportion_of_suspected_sam_cases_reported_in_village','proportion_of_diabetes_cases_reported_in_village','proportion_of_hypertension_cases_reported_in_village','proportion_of_anemic_cases_reported_in_village']]
df= pd.merge(df,child_sam, on="anganwadi_id", how='left')

df['high_risk_child_referred']=np.where((df['referral_facility_name_hr_1_child'].notnull()) | (df['referral_facility_name_hr_2_child'].notnull()) | (df['referral_facility_name_hr_3_child'].notnull()) | (df['referral_facility_name_hr_4_child'].notnull()),1,0)
df['dose_3_given']=np.where((df['vaccine_status_penta_3']=="Given"),1,0)

model = smf.logit("sam_confirmed ~high_risk_child_referred+anemia_2_trimester+child_linked_to_mother+dropout_child+average_call_duration_listened_child+proportion_of_anemic_cases_reported_in_village+proportion_of_confirmed_sam_cases_reported_in_village", data=df)

results = model.fit()
results.summary()


model_odds = pd.DataFrame(np.exp(results.params), columns= ['OR'])
#model_odds['z-value']= results.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(results.conf_int())
model_odds

decimals = 2

model_odds['OR'] =model_odds['OR'].apply(lambda x: round(x, decimals))
model_odds['2.5%'] =model_odds['2.5%'].apply(lambda x: round(x, decimals))
model_odds['97.5%'] =model_odds['97.5%'].apply(lambda x: round(x, decimals))
model_odds=model_odds[model_odds.index!="Intercept"]

writer = ExcelWriter('Child_Household_Visit_Malnutrition_Regression_Analysis.xlsx')
model_odds.to_excel(writer,'Child_Malnutrition_Regression',encoding='utf-8-sig',index=True)
writer.save()

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table  # EDIT: see deprecation warnings below

fig, ax = plt.subplots(figsize=(16, 1)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, model_odds, loc='upper right', colWidths=[0.17]*len(model_odds.columns))  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
#plt.savefig('table.png', transparent=True)# where df is your data frame
plt.title('Regression Model (Confirmed SAM Cases Against Household Visits)')
plt.savefig('Child_Household_Visit_Malnutrition_Regression_Analysis.png', transparent=False)



import seaborn as sns
sns.set_style('darkgrid')
import math
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_cols = ['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']

Xx = df[X_cols]


X_constant = sm.add_constant(Xx, prepend=False)


'''Linear assumptions '''
df_titanic_lt=df[ (df['average_call_duration_listened_child']>0) & (df['proportion_of_anemic_cases_reported_in_village']>0) & (df['proportion_of_confirmed_sam_cases_reported_in_village']>0)][['sam_confirmed','high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]

continuous_var = ['average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']
# Add logit transform interaction terms (natural log) for continuous variables e.g. Age * Log(Age)
for var in continuous_var:
    df_titanic_lt[f'{var}:Log_{var}'] = df_titanic_lt[var].apply(lambda x: x * np.log(x)) #np.log = natural log
df_titanic_lt.head()
# Keep columns related to continuous variables
cols_to_keep = continuous_var + df_titanic_lt.columns.tolist()[-len(continuous_var):]
# Redefine independent variables to include interaction terms
#df_titanic_lt['anm_average_data_quality_score']=df_titanic_lt['anm_average_data_quality_score']**5
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
#df_titanic_lt=df[X_cols]
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)

df_titanic_lt=pd.DataFrame((imp.fit_transform(df_titanic_lt)),columns =df_titanic_lt.columns)

X=(df_titanic_lt.copy())
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['sam_confirmed']]
# Add constant
X_lt_constant = sm.add_constant(X_lt, prepend=False) 
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt_constant, family=families.Binomial()).fit()
# Display summary results
print(logit_results.summary())

'''
#Polynomial Linear assumptions
df_titanic_lt=df[ (df['average_call_duration_listened_child']>0) & (df['proportion_of_anemic_cases_reported_in_village']>0) & (df['proportion_of_confirmed_sam_cases_reported_in_village']>0)][['sam_confirmed','high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]

continuous_var = ['average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']
# Add logit transform interaction terms (natural log) for continuous variables e.g. Age * Log(Age)
for var in continuous_var:
    df_titanic_lt[f'{var}:Log_{var}'] = df_titanic_lt[var].apply(lambda x: x * np.log(x)) #np.log = natural log
df_titanic_lt.head()
# Keep columns related to continuous variables
cols_to_keep = continuous_var + df_titanic_lt.columns.tolist()[-len(continuous_var):]
# Redefine independent variables to include interaction terms
#df_titanic_lt['anm_average_data_quality_score']=df_titanic_lt['anm_average_data_quality_score']**5
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
#df_titanic_lt=df[X_cols]
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)

df_titanic_lt=pd.DataFrame((imp.fit_transform(df_titanic_lt)),columns =df_titanic_lt.columns)

X=(df_titanic_lt.copy())
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['sam_confirmed']]
# Add constant
X_lt_constant = sm.add_constant(X_lt, prepend=False) 
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt_constant, family=families.Binomial()).fit()
# Display summary results
print(logit_results.summary())
'''

df_titanic_lt=df[['sam_confirmed','high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
#df_titanic_lt=df[X_cols]
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)



df_titanic_lt=pd.DataFrame((imp.fit_transform(df_titanic_lt)),columns =df_titanic_lt.columns)
cols_to_keep =['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['sam_confirmed']]

# Add constant
X_lt_constant = sm.add_constant(X_lt, prepend=False)
  
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()

# Display summary results
print(logit_results.summary())



# Use GLM method for logreg here so that we can retrieve the influence measures
logit_model = GLM(y_lt,X_lt, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())


# Re-run logistic regression on original set of X and y variables
#logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()
predicted = logit_results.predict(X_lt)

# Get log odds values
log_odds = np.log(predicted / (1 - predicted))

# Visualize predictor continuous variable vs logit values 
plt.scatter(x=X_lt['proportion_of_anemic_cases_reported_in_village'].values, y=log_odds);
plt.xlabel("Proportion of anemic cases reported in village")
plt.ylabel("Log-odds")
plt.show()


plt.scatter(x=X_lt['proportion_of_confirmed_sam_cases_reported_in_village'].values, y=log_odds);
plt.xlabel("Proportion of confirmed sam cases reported in village")
plt.ylabel("Log-odds")
plt.show()



# Visualize predictor continuous variable vs logit values 
plt.scatter(x=X_lt['average_call_duration_listened_child'].values, y=log_odds);
plt.xlabel("Average duration of educational reminder calls listened")
plt.ylabel("Log-odds")
plt.show()






# Visualize predictor continuous variable vs logit values 


from scipy import stats

# Get influence measures
influence = logit_results.get_influence()

# Obtain summary df of influence measures
summ_df = influence.summary_frame()

# Filter summary df to Cook distance
diagnosis_df = summ_df.loc[:,['cooks_d']]

# Append absolute standardized residual values
diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df.loc[:,'std_resid'].apply(lambda x: np.abs(x))

# Sort by Cook's Distance
diagnosis_df.sort_values("cooks_d", ascending=False)
diagnosis_df

# Set Cook's distance threshold
cook_threshold = 4 / len(X)
print(f"Threshold for Cook Distance = {cook_threshold}")

# Plot influence measures (Cook's distance)
fig = influence.plot_index(y_var="cooks", threshold=cook_threshold)
plt.axhline(y=cook_threshold, ls="--", color='red')
fig.tight_layout(pad=2)


# Find number of observations that exceed Cook's distance threshold
outliers = diagnosis_df[diagnosis_df['cooks_d'] > cook_threshold]
prop_outliers = round(100*(len(outliers) / len(X)),1)
print(f'Proportion of data points that are highly influential = {prop_outliers}%')


extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & 
                       (diagnosis_df['std_resid'] > 3)]
prop_extreme = round(100*(len(extreme) / len(X)),1)
print(f'Proportion of highly influential outliers = {prop_extreme}%')




corrMatrix = X_lt.corr()
plt.subplots(figsize=(10, 6))
sns.heatmap(corrMatrix, annot=True, cmap="RdYlGn")
plt.show()


# Use variance inflation factor to identify any significant multi-collinearity
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    decimals = 2

    vif["VIF"] =vif["VIF"].apply(lambda x: round(x, decimals))
    return(vif)

calc_vif(X_lt_constant)  # Include constant in VIF calculation in Python




# Setup logistic regression model (using GLM method so that we can retrieve residuals)
logit_model = GLM(y_lt, X_lt, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())

# Generate residual series plot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot", xlabel="Index Number", ylabel="Deviance Residuals")

# ax.plot(X.index.tolist(), stats.zscore(logit_results.resid_pearson))
ax.plot(X_lt.index.tolist(), stats.zscore(logit_results.resid_deviance))
plt.axhline(y=0, ls="--", color='red');

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(
            111,
            title="Residual Dependence Plot",
            xlabel="Fitted Values",
            ylabel="Pearson Residuals")

# ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_pearson))
ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_deviance))
ax.axis("tight")
ax.plot([0.0, 1.0], [0.0, 0.0], "k-");


#Setup LOWESS function
lowess = sm.nonparametric.lowess

# Get y-values from LOWESS (set return_sorted=False)
y_hat_lowess = lowess(logit_results.resid_pearson, logit_results.mu, 
                      return_sorted = False,
                      frac=2/3)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111,
    title="Residual Dependence Plot",
    xlabel="Fitted Values",
    ylabel="Pearson Residuals")

# ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_pearson))
ax.scatter(logit_results.mu, stats.zscore(logit_results.resid_deviance))
ax.scatter(logit_results.mu, y_hat_lowess)
ax.axis("tight")
ax.plot([0.0, 1.0], [0.0, 0.0], "k-");



import pandas as pd
import numpy as np
# importing the MICE from fancyimpute library
from fancyimpute import IterativeImputer
from fancyimpute import KNN 
from sklearn.preprocessing import OrdinalEncoder

anc=df[['sam_confirmed','high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]

NullValues=Xx.isnull().sum()*100/len(Xx)

mice_imputer = IterativeImputer()
# imputing the missing value with mice imputer
#df_lbw = mice_imputer.fit_transform(lbws)
encoder = OrdinalEncoder()
imputer = KNN()


anc = pd.DataFrame(np.round(imputer.fit_transform(anc)),columns = anc.columns)

''' shuffle the datafrmae'''    
anc = anc.sample(frac=1, random_state=42).reset_index(drop=True)    


import matplotlib.pyplot as plt







import matplotlib.pyplot as plt
X_lt = sm.add_constant(anc[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']], prepend=False)
y_lt=anc[['sam_confirmed']]
# Re-run logistic regression on original set of X and y variables
logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()
predicted = logit_results.predict(X_lt)

# Get log odds values
log_odds = np.log(predicted / (1 - predicted))



writercsv = os.path.join("sam_confirmed_assumptions_check.csv")
anc.to_csv(writercsv)

import numpy as np
np.random.seed(42)




''' training dataset'''
part_9 = anc.sample(frac = 0.8, random_state=2).reset_index(drop=True)    
''' test dataset'''
rest_part_25 = anc.drop(part_9.index)

logs_train=part_9.copy()
logs_test=rest_part_25.copy()


#logs_train['anm_average_data_quality_score']=logs_train['anm_average_data_quality_score']**5
#logs_test['anm_average_data_quality_score']=logs_test['anm_average_data_quality_score']**5




from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
X_train,X_test, y_train, y_test = train_test_split(logs_train[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']], logs_train['sam_confirmed'], test_size=0.1,  random_state=random.seed(1234))


np.random.seed(1234)
oversample = SMOTE()
''' over sampling of minority class'''
over_X, over_y = oversample.fit_resample(logs_train[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']], logs_train['sam_confirmed'])

''' split train test datset'''
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y, random_state=random.seed(1234))
#Build SMOTE SRF model

''' train the model using logistic regression approach'''
SMOTE_SRF =LogisticRegression(random_state=random.seed(1234))
'''Create Stratified K-fold cross validation and determine mean F1, recall and precision '''
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random.seed(1234))
scoring = ('f1', 'recall', 'precision')
#Evaluate SMOTE SRF model
scores = cross_validate(SMOTE_SRF, over_X_train, over_y_train, scoring=scoring, cv=cv)
#Get average evaluation metrics
print('Mean f1: %.3f' % mean(scores['test_f1']))
print('Mean recall: %.3f' % mean(scores['test_recall']))
print('Mean precision: %.3f' % mean(scores['test_precision']))


''' the below section is not required but just mentioned to see how the prediction on training dataset behaves alike'''
#Randomly spilt dataset to test and train set
X_train, X_test, y_train, y_test = train_test_split(logs_train[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']], logs_train['sam_confirmed'], test_size=0.2, stratify=logs_train['sam_confirmed'], random_state=random.seed(1234))


SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(logs_train[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']])
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, logs_train[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']],  logs_train['sam_confirmed'],display_labels=['Sam Confirmed', 'Not SAM'], cmap='Greens')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(logs_train['sam_confirmed'], y_pred)
print(cf_matrix)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(89)
import random as rn
rn.seed(1254)




#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X_test)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Sam Confirmed', 'Not SAM'], cmap='Reds')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

''' final confusion matrix '''
from sklearn.metrics import confusion_matrix
cnf_matrixl = confusion_matrix(y_test, y_pred)
print(cnf_matrixl)


''' plot the classification report'''
def confusion_metrics (conf_matrix):# save confusion matrix and slice into four pieces    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]    
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))   
    
    #calculate ppv & npv
    conf_ppv = (TP / int(TP + FP))
    conf_npv = (TN / int(TN + FN))
    FPR = (FP / float(FP + TN))
    FNR = (FN / float(FN + TP))
    
    LRP =conf_sensitivity/(1-conf_specificity)
    LRN =(1-conf_sensitivity)/(conf_specificity)
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Positive Predicted Value: {round(conf_ppv ,2)}')
    print(f'Negative Predicted Value: {round(conf_npv ,2)}')
    print(f'False Positive Rate: {round(FPR ,2)}')
    print(f'False Negative Rate: {round(FNR ,2)}')
    print(f'Likelihood Ratio Positive: {round(LRP ,2)}')
    print(f'Likelihood Ratio Negative: {round(LRN ,2)}')
    #print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')

print('logistic regression cm')
confusionl = confusion_metrics(cnf_matrixl)



from sklearn.utils import resample
from sklearn.utils import shuffle

for column in ['sam_confirmed','high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']:
    df[column] =df[column].fillna(df[column].mean())
    
df = df.sample(frac=1, random_state=42).reset_index(drop=True)    



import numpy as np
np.random.seed(42)


part_9 = df.sample(frac = 0.8, random_state=2).reset_index(drop=True)     

rest_part_25 = df.drop(part_9.index)




from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

#Use SMOTE to oversample the minority class
oversample = SMOTE()
y=part_9[['sam_confirmed']]

X=part_9[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]
training_features=X.copy()
training_target=y.copy()
np.random.seed(1234)
over_X, over_y = oversample.fit_resample(X, y)
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y, random_state=random.seed(1234))
#Build SMOTE SRF model
SMOTE_SRF = RandomForestClassifier(n_estimators=300,max_depth=10,random_state=random.seed(1234))
#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random.seed(1234))
scoring = ('f1', 'recall', 'precision')
#Evaluate SMOTE SRF model
scores = cross_validate(SMOTE_SRF, over_X_train, over_y_train, scoring=scoring, cv=cv)
#Get average evaluation metrics
print('Mean f1: %.3f' % mean(scores['test_f1']))
print('Mean recall: %.3f' % mean(scores['test_recall']))
print('Mean precision: %.3f' % mean(scores['test_precision']))


#Randomly spilt dataset to test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random.seed(1234))
#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X ,y, display_labels=['Not SAM', 'SAM'], cmap='Greens')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y, y_pred)
print(cf_matrix)



random.seed(42)
y_test=rest_part_25[['sam_confirmed']]

X_test=rest_part_25[['high_risk_child_referred','anemia_2_trimester','child_linked_to_mother','dropout_child','average_call_duration_listened_child','proportion_of_anemic_cases_reported_in_village','proportion_of_confirmed_sam_cases_reported_in_village']]


#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X_test)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Not SAM', 'SAM'], cmap='Reds')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

#explainer = ClassifierExplainer(model, X_test, y_test)
#ExplainerDashboard(explainer).run()
from explainerdashboard import *
from sklearn.neural_network import MLPClassifier
from explainerdashboard.datasets import *

#X_train, y_train, X_test, y_test = titanic_survive()
model = MLPClassifier().fit(X_train, y_train)
explainer = ClassifierExplainer(model, X_test, y_test, shap='kernel')   
ExplainerDashboard(explainer, shap_interaction=False).run()


from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, titanic_names

feature_descriptions = {
    "Sex": "Gender of passenger",
    "Gender": "Gender of passenger",
    "Deck": "The deck the passenger had their cabin on",
    "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
    "Fare": "The amount of money people paid", 
    "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
    "Age": "Age of the passenger",
    "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
    "No_of_parents_plus_children_on_board" : "The sum of the number of parents plus the number of children on board",
}

X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                                cats=['Deck', 'Embarked',
                                    {'Gender': ['Sex_male', 'Sex_female', 'Sex_nan']}],
                                cats_notencoded={'Embarked': 'Stowaway'}, # defaults to 'NOT_ENCODED'
                                descriptions=feature_descriptions, # adds a table and hover labels to dashboard
                                labels=['Not survived', 'Survived'], # defaults to ['0', '1', etc]
                                idxs = test_names, # defaults to X.index
                                index_name = "Passenger", # defaults to X.index.name
                                target = "Survival", # defaults to y.name
                                )

db = ExplainerDashboard(explainer, 
                        title="Titanic Explainer", # defaults to "Model Explainer"
                        shap_interaction=False, # you can switch off tabs with bools
                        )
db.run(port=8050)

 
''' Deep Neural network with 5 layers '''

from sklearn.utils import resample

np.random.seed(42)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
''' Deep Neural network with 5 layers '''
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
tf.random.set_seed(89)
import random as rn
rn.seed(1254)
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=random.seed(1234))


sm = SMOTE(random_state=8, sampling_strategy= 1)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)



model_1 = Sequential(random.seed(1234))
model_1.add(Dense(80, input_dim=7, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(80, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(80, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dense(1, activation='sigmoid'))



model_1.compile(optimizer = 'adam',
           loss = 'binary_crossentropy',
           metrics=['accuracy'])



model_1.fit(x_train_res, y_train_res , epochs=1500)
preds = model_1.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(y_test,preds)
auc_score3 = auc(false_positive_rate, recall)




import seaborn as sns
from sklearn.metrics import confusion_matrix
cnf_matrix_n=confusion_matrix(y_test, preds.round())

confusion=confusion_metrics(cf_matrix)


print('random forest cm')
confusionn=confusion_metrics(cf_matrix)
print('logistic regression cm')
confusionl = confusion_metrics(cnf_matrixl)

print('neural network cm')
confusionnn=confusion_metrics(cnf_matrix_n)
''' plot confusion matrix'''
import seaborn as sns

import seaborn as sns

a= (cf_matrix)[0][0]
b=(cf_matrix)[1][1]
(cf_matrix)[0][0]=b
(cf_matrix)[1][1]=a
group_names = ['','','','']
#zs=np.array([48, 179,  13,  958])
group_names = ['','','','']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)
#zs= np.reshape(zs, (2, 2))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')

ax.set_title('Confirmed SAM Cases - Random Forest');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Confirmed SAM Cases (Positive)','Normal (Negative)'])
ax.yaxis.set_ticklabels(['Positive','Negative'])


''' Plot ROC Curve (Logistic regression versus random forest - smoote)'''

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train=X.copy()
y_train=y.copy()
model1 = LogisticRegression()
# knn
model2 = RandomForestClassifier(n_estimators=1000,max_depth=20,random_state=random.seed(1234))
# fit model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization



''' plot confusion matrix'''
import seaborn as sns

a= (cnf_matrix_n)[0][0]
b=(cnf_matrix_n)[1][1]
(cnf_matrix_n)[0][0]=b
(cnf_matrix_n)[1][1]=a
group_names = ['','','','']
#zs=np.array([48, 179,  13,  958])
group_names = ['','','','']
group_counts = ["{0:0.0f}".format(value) for value in
               cnf_matrix_n.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cnf_matrix_n.flatten()/np.sum(cnf_matrix_n)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)
#zs= np.reshape(zs, (2, 2))
ax = sns.heatmap(cnf_matrix_n, annot=labels, fmt='', cmap='Reds')

ax.set_title('Confirmed SAM Cases - Deep Neural Network');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Confirmed SAM Cases (Positive)','Normal (Negative)'])
ax.yaxis.set_ticklabels(['Positive','Negative'])

# fit model
model1.fit(over_X_train, over_y_train)
model2.fit(over_X_train, over_y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict_proba(X_test)
pred_prob3 = model_1.predict(X_test)


random_probs = [0 for i in range(len(over_y_train))]
p_fpr, p_tpr, _ = roc_curve(over_y_train, random_probs, pos_label=1)


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:, 1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:, 1])
print(auc_score1, auc_score2,auc_score3)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3  = roc_curve(y_test,preds)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


from sklearn.metrics import roc_auc_score

# auc scores
#auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
#auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])


#print(auc_score1, auc_score2)
#pred_prob11 = model1.predict(X_test)
#pred_prob22 = model2.predict(X_test)


from sklearn.metrics import confusion_matrix
#cf_matrix = confusion_matrix(y_test, pred_prob22)
#print(cf_matrix)


import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='-',color='red', label='Logistic Regression (area = %0.2f)' % auc_score1)
plt.plot(fpr2, tpr2, linestyle='--',color='green',  label='Random Forest (area = %0.2f)' % auc_score2)
plt.plot(fpr3, tpr3, linestyle='--',color='black',  label='Deep Neural Network With 2 Hidden Layers  (area = %0.2f)' % auc_score3)
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC Curve - SAM against household visit model')

# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
#plt.plot(fpr1, tpr1, color='red',label='Logistic Regression (area = %0.2f)' % auc_score1)
#plt.plot(fpr2, tpr2,color='green', label='Random Forest (area = %0.2f)' % auc_score2)
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)

plt.show();





def sensitivity_check(training_features,training_target,sampling_ratio,ax_title,ax_tick_labels):
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    np.random.seed(42)
    tf.random.set_seed(89)
    import random as rn
    rn.seed(1254)
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                      test_size = .1,random_state=random.seed(1234))


    sm = SMOTE(random_state=12, sampling_strategy= sampling_ratio)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score,precision_score,accuracy_score
    from imblearn.over_sampling import SMOTE

    clf_rf = RandomForestClassifier(n_estimators=300,max_depth=10,random_state=random.seed(1234))
    clf_rf.fit(x_train_res, y_train_res)
    print('Validation Results')
    print(clf_rf.score(x_val, y_val))
    print(recall_score(y_val, clf_rf.predict(x_val)))
    print('\nTest Results')
    print(clf_rf.score(X_test, y_test))
    print('Recall - Test Results')
    print(recall_score(y_test, clf_rf.predict(X_test)))
    print('Precision - Test Results')
    print(precision_score(y_test, clf_rf.predict(X_test)))
    print('Accuracy - Test Results')
    print(accuracy_score(y_test, clf_rf.predict(X_test)))
    #plot_confusion_matrix(confusion_matrix(y_test,clf_rf.predict(X_test)))

    cf_matrix=confusion_matrix(y_test,clf_rf.predict(X_test))
    
    ''' plot confusion matrix'''
    import seaborn as sns

    a= (cf_matrix)[0][0]
    b=(cf_matrix)[1][1]
    (cf_matrix)[0][0]=b
    (cf_matrix)[1][1]=a
    print(cf_matrix)
    group_names = ['','','','']
    #zs=np.array([48, 179,  13,  958])
    group_names = ['','','','']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    #zs= np.reshape(zs, (2, 2))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')

    ax.set_title(ax_title);
    ax.set_xlabel('\nThe Truth')
    ax.set_ylabel('Test Score ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(ax_tick_labels)
    #ax.xaxis.set_ticklabels(['SAM (Positive)','Not SAM (Negative)'])
    ax.yaxis.set_ticklabels(['Positive','Negative'])

rf_60=sensitivity_check(training_features,training_target,0.6,'Confirmed SAM Cases - Random Forest (60% Over Sampling)',['SAM (Positive)','Not SAM (Negative)'])    
rf_70=sensitivity_check(training_features,training_target,0.7,'Confirmed SAM Cases - Random Forest (70% Over Sampling)',['SAM (Positive)','Not SAM (Negative)'])
rf_75=sensitivity_check(training_features,training_target,0.75,'Confirmed SAM Cases - Random Forest (75% Over Sampling)',['SAM (Positive)','Not SAM (Negative)'])
rf_90=sensitivity_check(training_features,training_target,0.9,'Confirmed SAM Cases - Random Forest (90% Over Sampling)',['SAM (Positive)','Not SAM (Negative)'])







import shap
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Fits the explainer
explainer = shap.Explainer(SMOTE_SRF.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)
# Calculates the SHAP values - It takes some time

# Evaluate SHAP values
#shap_values = explainer.shap_values(X)

shap.plots.bar(shap_values)
shap.summary_plot(shap_values)
shap.summary_plot(shap_values, plot_type='violin')
shap.plots.bar(shap_values[0])

shap.plots.force(shap_values[0])






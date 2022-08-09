

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 01:43:06 2022

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
import statistics
import statsmodels.formula.api as smf
from scipy import stats
import datetime
random.seed(42)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
''' Deep Neural network with 5 layers 
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
tf.random.set_seed(89)
import random as rn
rn.seed(1254)
'''
os.chdir("/home/sarfraaz/Videos/kb_smu_child_malnutrition")
df = os.path.join("live_still_birth2.csv")
df = pd.read_csv(df)

df2 = os.path.join("all_scores2.csv")
df2 = pd.read_csv(df2)
df2['Clusters']=df2['Clusters'].replace(0,10)
df2['Clusters']=df2['Clusters'].replace(1,7)
df2['Clusters']=df2['Clusters'].replace(2,5)
df2['Clusters']=df2['Clusters'].replace(3,3)

df2['anm_average_data_quality_score']=df2.groupby(['sub_center_id'])['Scores'].transform('mean')
df2['type_of_performer_since_registration']=df2.groupby(['sub_center_id'])['Clusters'].transform('mean')

last_6_month=df2['Months_id'].max()-6
last_6=df2[df2['Months_id']>=last_6_month]

last_6['anm_average_data_quality_score_last_6_month']=last_6.groupby(['sub_center_id'])['Scores'].transform('mean')
last_6['type_of_performer_since_registration_last_6_month']=last_6.groupby(['sub_center_id'])['Clusters'].transform('mean')



last_6=last_6.sort_values(['sub_center_id','Months_id'], ascending=True)

last_6=(last_6.drop_duplicates(subset=['sub_center_id'], keep='last'))[['sub_center_id','anm_average_data_quality_score_last_6_month','type_of_performer_since_registration_last_6_month']]

df2= pd.merge(df2,last_6, on="sub_center_id", how='left')
df2=df2.sort_values(['sub_center_id','Months_id'], ascending=True)

df2=(df2.drop_duplicates(subset=['sub_center_id'], keep='last'))
df= pd.merge(df,df2, on="sub_center_id", how='left')
df['anemia_1']=np.where((df['hb_1']<=8),1,0)
df['anemia_2']=np.where((df['hb_2']<=8),1,0)
df['anemia_3']=np.where((df['hb_3']<=8),1,0)
df['anemia_4']=np.where((df['hb_4']<=8),1,0)

df['anc_date_1'] = pd.to_datetime(df['anc_date_1'], errors='coerce').dt.date
df['anc_date_2'] = pd.to_datetime(df['anc_date_2'], errors='coerce').dt.date
df['anc_date_3'] = pd.to_datetime(df['anc_date_3'], errors='coerce').dt.date
df['anc_date_4'] = pd.to_datetime(df['anc_date_4'], errors='coerce').dt.date
df['lmp_date'] = pd.to_datetime(df['lmp_date'], errors='coerce').dt.date
df['edd']=df['lmp_date']+timedelta(280)

df['first_diff'] = (df['anc_date_1']-df['lmp_date']).dt.days
df['second_diff'] = (df['anc_date_2']-df['lmp_date']).dt.days
df['third_diff'] = (df['anc_date_3']-df['lmp_date']).dt.days
df['fourth_diff'] = (df['anc_date_4']-df['lmp_date']).dt.days    

df['anemia_1_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']<=90)) | ((df['hb_2']<=8) & (df['first_diff']<=90)) | ((df['hb_3']<=8) & (df['first_diff']<=90)) | ((df['hb_4']<=8) & (df['first_diff']<=90)) | ((df['hb_1']<=8) & (df['second_diff']<=90)) | ((df['hb_2']<=8) & (df['second_diff']<=90)) | ((df['hb_3']<=8) & (df['second_diff']<=90)) | ((df['hb_4']<=8) & (df['second_diff']<=90)) | ((df['hb_1']<=8) & (df['third_diff']<=90)) | ((df['hb_2']<=8) & (df['third_diff']<=90)) | ((df['hb_3']<=8) & (df['third_diff']<=90)) | ((df['hb_4']<=8) & (df['third_diff']<=90)),1,0)
df['anemia_2_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_2']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_3']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_4']<=8) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['hb_1']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_2']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_3']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_4']<=8) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['hb_1']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_2']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_3']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['hb_4']<=8) & (df['third_diff']>90) & (df['third_diff']<=180)),1,0)
df['anemia_3_trimester']=np.where(((df['hb_1']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_2']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_3']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_4']<=8) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['hb_1']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_2']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_3']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_4']<=8) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['hb_1']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_2']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_3']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['hb_4']<=8) & (df['third_diff']>180) & (df['third_diff']<=340)),1,0)

df['hypertension_1_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']<=90)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']<=90)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']<=90)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']<=90)),1,0)
df['hypertension_2_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']>90) & (df['first_diff']<=180)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']>90) & (df['second_diff']<=180)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']>90) & (df['third_diff']<=180)),1,0)
df['hypertension_3_trimester']=np.where(((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['first_diff']>180) & (df['first_diff']<=340)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['second_diff']>180) & (df['second_diff']<=340)) | ((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)) | ((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140) & (df['third_diff']>180) & (df['third_diff']<=340)),1,0)

df['hypertension_1']=np.where((df['bp_dia_1']>=90) | (df['bp_sys_1']>=140),1,0)
df['hypertension_2']=np.where((df['bp_dia_2']>=90) | (df['bp_sys_2']>=140),1,0)
df['hypertension_3']=np.where((df['bp_dia_3']>=90) | (df['bp_sys_3']>=140),1,0)
df['hypertension_4']=np.where((df['bp_dia_4']>=90) | (df['bp_sys_4']>=140),1,0)

df['shypertension_1']=np.where((df['bp_dia_1']>=110) | (df['bp_sys_1']>=160),1,0)
df['shypertension_2']=np.where((df['bp_dia_2']>=110) | (df['bp_sys_2']>=160),1,0)
df['severe_hypertension_3']=np.where((df['bp_dia_3']>=110) | (df['bp_sys_3']>=160),1,0)
df['shypertension_4']=np.where((df['bp_dia_4']>=110) | (df['bp_sys_4']>=160),1,0)

df['high_risk_1']=np.where((df['referral_facility_name_hr_1'].notnull()) ,1,0)
df['high_risk_2']=np.where((df['referral_facility_name_hr_2'].notnull()) ,1,0)
df['high_risk_3']=np.where((df['referral_facility_name_hr_3'].notnull()) ,1,0)
df['high_risk_4']=np.where((df['referral_facility_name_hr_4'].notnull()) ,1,0)


df['high_risk_mother']=np.where((df['high_risk_1']>0) | (df['high_risk_2']>0) | (df['high_risk_3']>0) | (df['high_risk_4']>0) ,1,0)

df['four_anc_completed']=np.where((df['total_anc_completed']>=4),1,0)


df['mother_age_less_than_18']=np.where((df['mother_age']<18),1,0)  
df['mother_age_greater_than_35']=np.where((df['mother_age']>=35),1,0) 
df['pregnancy_greater_than_3']=np.where((df['pregnancy_no']>3),1,0) 
df['pregnancy_greater_than_2']=np.where((df['pregnancy_no']>2),1,0) 
df['pregnancy_greater_than_4']=np.where((df['pregnancy_no']>4),1,0) 
df['tt_booster_given']=np.where((df['tt_booster_date'].notnull()),1,0) 
df['tt_1_given']=np.where((df['tt1_date'].notnull()),1,0) 
df['tt_2_given']=np.where((df['tt2_date'].notnull()),1,0) 

df['low_height']=np.where((df['height']<140),1,0) 
df['mother_age_greater_than_20_less_than_25']=np.where((df['mother_age']>=20) & (df['mother_age']<25),1,0) 
df['mother_age_greater_than_25_less_than_30']=np.where((df['mother_age']>=25) & (df['mother_age']<30),1,0) 
df['mother_age_greater_than_30_less_than_35']=np.where((df['mother_age']>=30) & (df['mother_age']<35),1,0) 
df['mother_age_greater_than_20_less_than_30']=np.where((df['mother_age']>=20) & (df['mother_age']<30),1,0) 

df.live_still_birth=df.live_still_birth.replace(1,2)
df.live_still_birth=df.live_still_birth.replace(0,1)
df.live_still_birth=df.live_still_birth.replace(2,0)
model = smf.logit("live_still_birth~weight_1+severe_hypertension_3+bp_sys_1+mother_age_greater_than_25_less_than_30+four_anc_completed+anm_average_data_quality_score+days_to_edd_from_last_checkup", data=df)

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

writer = ExcelWriter('Live_Still_Birth_Regression_Analysis.xlsx')
model_odds.to_excel(writer,'Live_Still_Birth_Regression',encoding='utf-8-sig',index=True)
writer.save()
model_odds['97.5%']=np.where((model_odds['2.5%']<1) & (model_odds['97.5%']==1),0.99,model_odds['97.5%'])
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
plt.title('Regression Model (Still Birth)')
plt.savefig('Live_Still_Birth_Regression_Analysis.png', transparent=False)


import seaborn as sns
sns.set_style('darkgrid')
import math
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_cols = ['weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']
Xx = df[X_cols]


X_constant = sm.add_constant(Xx, prepend=False)
'''Linear assumptions '''
df_titanic_lt=df[(df['bp_sys_1']>0) & (df['weight_1']>0) & (df['anm_average_data_quality_score']>0) & (df['days_to_edd_from_last_checkup']>0)][['weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup','live_still_birth']]
# Define continuous variables
continuous_var = ['bp_sys_1', 'weight_1','anm_average_data_quality_score','days_to_edd_from_last_checkup']
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
X=imp.fit_transform(df_titanic_lt)
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['live_still_birth']]
# Add constant
X_lt_constant = sm.add_constant(X_lt, prepend=False) 
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt_constant, family=families.Binomial()).fit()
# Display summary results
print(logit_results.summary())


X_cols = ['weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']
Xx = df[X_cols]
X_constant = sm.add_constant(Xx, prepend=False)
df_titanic_lt=df[(df['bp_sys_1']>0) & (df['weight_1']>0) & (df['anm_average_data_quality_score']>0) & (df['days_to_edd_from_last_checkup']>0)][['weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup','live_still_birth']]
# Define continuous variables
continuous_var = ['bp_sys_1', 'weight_1','anm_average_data_quality_score','days_to_edd_from_last_checkup']
# Add logit transform interaction terms (natural log) for continuous variables e.g. Age * Log(Age)
for var in continuous_var:
    df_titanic_lt[f'{var}:Log_{var}'] = df_titanic_lt[var].apply(lambda x: x * np.log(x)) #np.log = natural log
df_titanic_lt.head()
# Keep columns related to continuous variables
cols_to_keep = continuous_var + df_titanic_lt.columns.tolist()[-len(continuous_var):]
# Redefine independent variables to include interaction terms
df_titanic_lt['anm_average_data_quality_score']=df_titanic_lt['anm_average_data_quality_score']**5
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
#df_titanic_lt=df[X_cols]
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)
X=imp.fit_transform(df_titanic_lt)
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['live_still_birth']]
# Add constant
X_lt_constant = sm.add_constant(X_lt, prepend=False) 
# Build model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt_constant, family=families.Binomial()).fit()
# Display summary results
print(logit_results.summary())

df_titanic_lt=df[['live_still_birth','weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']]

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
#df_titanic_lt=df[X_cols]
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)
X=imp.fit_transform(df_titanic_lt)



df_titanic_lt=pd.DataFrame((imp.fit_transform(df_titanic_lt)),columns =df_titanic_lt.columns)
cols_to_keep =['weight_1','severe_hypertension_3','bp_sys_1','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']
X_lt = df_titanic_lt[cols_to_keep]
y_lt = df_titanic_lt[['live_still_birth']]

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
logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()
predicted = logit_results.predict(X_lt)

# Get log odds values
log_odds = np.log(predicted / (1 - predicted))

# Visualize predictor continuous variable vs logit values 
plt.scatter(x=X_lt['anm_average_data_quality_score'].values, y=log_odds);
plt.xlabel("ANM average data quality score")
plt.ylabel("Log-odds")
plt.show()


# Visualize predictor continuous variable vs logit values 
plt.scatter(x=X_lt['bp_sys_1'].values, y=log_odds);
plt.xlabel("Systolic blood pressure during ANC 1")
plt.ylabel("Log-odds")
plt.show()


# Visualize predictor continuous variable vs logit values 
plt.scatter(x=X_lt['weight_1'].values, y=log_odds);
plt.xlabel("Weight during ANC 1")
plt.ylabel("Log-odds")
plt.show()


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

anc=df[['live_still_birth','weight_1','severe_hypertension_3','bp_sys_1','days_to_anc_1_from_lmp','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']]

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
X_lt = sm.add_constant(anc[['weight_1','severe_hypertension_3','bp_sys_1','days_to_anc_1_from_lmp','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']], prepend=False)
y_lt=anc[['live_still_birth']]
# Re-run logistic regression on original set of X and y variables
logit_results = GLM(y_lt, X_lt, family=families.Binomial()).fit()
predicted = logit_results.predict(X_lt)

# Get log odds values
log_odds = np.log(predicted / (1 - predicted))



writercsv = os.path.join("live_still_birth_assumptions_check.csv")
anc.to_csv(writercsv)

import numpy as np
np.random.seed(42)




''' training dataset'''
part_9 = anc.sample(frac = 0.8, random_state=2).reset_index(drop=True)    
''' test dataset'''
rest_part_25 = anc.drop(part_9.index)

logs_train=part_9.copy()
logs_test=rest_part_25.copy()


logs_train['anm_average_data_quality_score']=logs_train['anm_average_data_quality_score']**5
logs_test['anm_average_data_quality_score']=logs_test['anm_average_data_quality_score']**5




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
X_train,X_test, y_train, y_test = train_test_split(logs_train[[ 'weight_1', 'severe_hypertension_3', 'bp_sys_1',
       'days_to_anc_1_from_lmp', 'mother_age_greater_than_25_less_than_30',
       'four_anc_completed', 'anm_average_data_quality_score',
       'days_to_edd_from_last_checkup']], logs_train['live_still_birth'], test_size=0.1,  random_state=random.seed(1234))


np.random.seed(1234)
oversample = SMOTE()
''' over sampling of minority class'''
over_X, over_y = oversample.fit_resample(logs_train[[ 'weight_1', 'severe_hypertension_3', 'bp_sys_1',
       'days_to_anc_1_from_lmp', 'mother_age_greater_than_25_less_than_30',
       'four_anc_completed', 'anm_average_data_quality_score',
       'days_to_edd_from_last_checkup']], logs_train['live_still_birth'])

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
X_train, X_test, y_train, y_test = train_test_split(logs_train[[ 'weight_1', 'severe_hypertension_3', 'bp_sys_1',
       'days_to_anc_1_from_lmp', 'mother_age_greater_than_25_less_than_30',
       'four_anc_completed', 'anm_average_data_quality_score',
       'days_to_edd_from_last_checkup']], logs_train['live_still_birth'], test_size=0.2, stratify=logs_train['live_still_birth'], random_state=random.seed(1234))


SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(logs_train[[ 'weight_1', 'severe_hypertension_3', 'bp_sys_1',
       'days_to_anc_1_from_lmp', 'mother_age_greater_than_25_less_than_30',
       'four_anc_completed', 'anm_average_data_quality_score',
       'days_to_edd_from_last_checkup']])
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, logs_train[[ 'weight_1', 'severe_hypertension_3', 'bp_sys_1',
       'days_to_anc_1_from_lmp', 'mother_age_greater_than_25_less_than_30',
       'four_anc_completed', 'anm_average_data_quality_score',
       'days_to_edd_from_last_checkup']], logs_train['live_still_birth'], display_labels=['Live Birth', 'Still Birth'], cmap='Greens')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(logs_train['live_still_birth'], y_pred)
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
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Live Birth', 'Still Birth'], cmap='Reds')
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
#cnf_matrixl = metrics.confusion_matrix(y_test, y_predl)





#Use SMOTE to oversample the minority class
oversample = SMOTE()
y=part_9[['live_still_birth']]

X=part_9[['weight_1','severe_hypertension_3','bp_sys_1','days_to_anc_1_from_lmp','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']]

training_features=X.copy()
training_target=y.copy()
np.random.seed(1234)
''' over sampling of minority class'''
over_X, over_y = oversample.fit_resample(X, y)

''' split train test datset'''
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y, random_state=random.seed(1234))
#Build SMOTE SRF model

''' train the model using random forest approach'''
SMOTE_SRF = RandomForestClassifier(n_estimators=1000,max_depth=20,random_state=random.seed(1234))
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random.seed(1234))
#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X ,y, display_labels=['Live Birth', 'Still Birth'], cmap='Greens')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y, y_pred)
print(cf_matrix)




''' test the result on 10% unseen dataset'''
y_test=rest_part_25[['live_still_birth']]

X_test=rest_part_25[['weight_1','severe_hypertension_3','bp_sys_1','days_to_anc_1_from_lmp','mother_age_greater_than_25_less_than_30','four_anc_completed','anm_average_data_quality_score','days_to_edd_from_last_checkup']]



#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X_test)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Live Birth', 'Still Birth'], cmap='Reds')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()

''' final confusion matrix '''
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)




 
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
model_1.add(Dense(80, input_dim=8, activation='relu'))
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

ax.set_title('Still Birth model - Random Forest');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Still Birth (Positive)','Live Birth (Negative)'])
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

ax.set_title('Still Birth Model - Deep Neural Network');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Still Birth (Positive)','Live Birth (Negative)'])
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
plt.title('ROC Curve - Still Birth model')

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

    clf_rf = RandomForestClassifier(n_estimators=1000,max_depth=20,random_state=random.seed(1234))
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
    
rf_10=sensitivity_check(training_features,training_target,0.1,'Still Birth Model - Random Forest (10% Over Sampling)',['Still birth (Positive)','Live birth (Negative)'])
rf_30=sensitivity_check(training_features,training_target,0.3,'Still Birth Model - Random Forest (30% Over Sampling)',['Still birth (Positive)','Live birth (Negative)'])
rf_50=sensitivity_check(training_features,training_target,0.5,'Still Birth Model - Random Forest (50% Over Sampling)',['Still birth (Positive)','Live birth (Negative)'])
rf_70=sensitivity_check(training_features,training_target,0.7,'Still Birth Model - Random Forest (70% Over Sampling)',['Still birth (Positive)','Live birth (Negative)'])
rf_90=sensitivity_check(training_features,training_target,0.9,'Still Birth Model - Random Forest (90% Over Sampling)',['Still birth (Positive)','Live birth (Negative)'])
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
#create shap plots
shap.plots.bar(shap_values)
shap.summary_plot(shap_values)
shap.summary_plot(shap_values, plot_type='violin')
shap.plots.bar(shap_values[0])

shap.plots.force(shap_values[0])

'''

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,random_state=random.seed(1234))



#Train the classifier.
bbc.fit(training_features, training_target)
preds = bbc.predict(X_test)
cf_matrix=confusion_matrix(y_test,bbc.predict(X_test))







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

ax.set_title('Still birth model - Balanced Bagging Classifier');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Still birth (Positive)','Live birth (Negative)'])
ax.yaxis.set_ticklabels(['Positive','Negative'])










np.random.seed(1234)
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












import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(training_features, training_target)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random.seed(1234))
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
clf = SelfPacedEnsembleClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict with an SPE classifier
y_pred = clf.predict(X_test)


features_train=np.array(X_train)
target_train=np.array(y_train)
features_test=np.array(X_test)
target_test=np.array(y_test)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
#Deep Neural network with 5 layers '
model_1 = Sequential()
model_1 .add(Dense(80, input_dim=11, activation='relu'))
model_1 .add(Dropout(0.2))
model_1 .add(Dense(80, activation='relu'))
model_1 .add(Dropout(0.2))
model_1 .add(Dense(80, activation='relu'))
model_1 .add(BatchNormalization())
model_1 .add(Dense(1, activation='sigmoid'))



model_1.compile(optimizer = 'adam',
           loss = 'binary_crossentropy',
           metrics=['accuracy'])



model_1.fit(features_train_scaled, target_train, epochs=30)
preds = model_1.predict(features_test_scaled)

test_loss, test_acc = model_1.evaluate(features_test_scaled,target_test)

print('Test accuracy:', test_acc)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(target_test, preds.round())
cnf_matrix
ax= plt.subplot()
sns.heatmap(cnf_matrix, annot=True,cmap=plt.cm.Blues, ax = ax, fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('        Confusion Matrix - Deep Neural network with 5 layers'); 
ax.xaxis.set_ticklabels((['0', '1'])); ax.yaxis.set_ticklabels(['0', '1']);
plt.savefig("Confusion Matrix DNN 5 Layers.png")



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

weights = np.linspace(0.05, 0.95, 20)

gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, y)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from matplotlib import pyplot as plt

lr = LogisticRegression(**grid_result.best_params_)

# Fit..
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred))

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

# Fit..
pipe.fit(X_train, y_train)

# Predict..
y_pred = pipe.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred))


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pipe = make_pipeline(
    SMOTE(),
    LogisticRegression())

weights = np.linspace(0.005, 0.05, 10)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={'smote__ratio': weights
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, y)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')

















from imblearn.over_sampling import ADASYN

X_resampled, y_resampled = ADASYN().fit_resample(X, y)

#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X_test)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Live Birth', 'Still Birth'], cmap='Reds')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show()


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(random_state=random.seed(1234))
# fit the model with data
logreg.fit(over_X_train, over_y_train)
#predict
y_predl=logreg.predict(X_test)


#confusion matrix
cnf_matrixl = metrics.confusion_matrix(y_test, y_predl)


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
#create shap plots
shap.plots.bar(shap_values)
shap.summary_plot(shap_values)
shap.summary_plot(shap_values, plot_type='violin')
shap.plots.bar(shap_values[0])

shap.plots.force(shap_values[0])

#plot the classification report
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
    conf_ppv = (TP / float(TP + FP))
    conf_npv = (TN / float(TN + FN))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Positive Predicted Value: {round(conf_ppv ,2)}')
    print(f'Negative Predicted Value: {round(conf_npv ,2)}')
    
    #print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')
 
confusion=confusion_metrics(cf_matrix)
confusionl=confusion_metrics(cnf_matrixl)    
#plot confusion matrix
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

ax.set_title('Still birth model -  SMOTE + Standard Random Forest Confusion Matrix');
ax.set_xlabel('\nThe Truth')
ax.set_ylabel('Test Score ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Still birth (Positive)','Live birth (Negative)'])
ax.yaxis.set_ticklabels(['Positive','Negative'])


#Plot ROC Curve (Logistic regression versus random forest - smoote)

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
model1.fit(over_X_train, over_y_train)
model2.fit(over_X_train, over_y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict_proba(X_test)


random_probs = [0 for i in range(len(over_y_train))]
p_fpr, p_tpr, _ = roc_curve(over_y_train, random_probs, pos_label=1)


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:, 1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:, 1])

print(auc_score1, auc_score2)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])


print(auc_score1, auc_score2)
pred_prob11 = model1.predict(X_test)
pred_prob22 = model2.predict(X_test)


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, pred_prob22)
print(cf_matrix)


import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='-',color='red', label='Logistic Regression (area = %0.2f)' % auc_score1)
plt.plot(fpr2, tpr2, linestyle='--',color='green',  label='SMOTE + Standard Random Forest (area = %0.2f)' % auc_score2)
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC Curve (SMOTE + Standard Random Forest) - Still birth model')

# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
#plt.plot(fpr1, tpr1, color='red',label='Logistic Regression (area = %0.2f)' % auc_score1)
#plt.plot(fpr2, tpr2,color='green', label='Random Forest (area = %0.2f)' % auc_score2)
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)

plt.show();







from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Model and performance
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Ensembled sampling
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
###### Step 4: Baseline Model
# Train the random forest model using the imbalanced dataset
rf = RandomForestClassifier()
baseline_model_cv = cross_validate(rf, X_train, y_train, cv = 5, n_jobs = None, scoring="recall")
# Check the model performance
print(f"{baseline_model_cv['test_score'].mean():.3f} +/- {baseline_model_cv['test_score'].std():.3f}")


###### Step 5: Balanced Random Forest Classifier
# Train the balanced random forest model
brf = BalancedRandomForestClassifier(random_state=42)
brf_model_cv = cross_validate(brf, X_train, y_train, cv = 10, scoring="recall")
# Check the model performance
print(f"{brf_model_cv['test_score'].mean():.3f} +/- {brf_model_cv['test_score'].std():.3f}")

###### Step 6: Random Under-Sampling Boosting Classifier
# Train the random under-sampling boosting classifier model
rusb = RUSBoostClassifier(random_state=42)
rusb_model_cv = cross_validate(rusb, X_train, y_train, cv = 5, n_jobs = -1, scoring="recall")
# Check the model performance
print(f"{rusb_model_cv['test_score'].mean():.3f} +/- {rusb_model_cv['test_score'].std():.3f}")


###### Step 7: Easy Ensemble Classifier for Ada Boost Classifier
# Train the easy ensemble classifier model
eec = EasyEnsembleClassifier(random_state=42)
eec_model_cv = cross_validate(eec, X_train, y_train, cv = 5, n_jobs = -1, scoring="recall")
# Check the model performance
print(f"{eec_model_cv['test_score'].mean():.3f} +/- {eec_model_cv['test_score'].std():.3f}")

###### Step 8: Balanced Bagging Classifier - Near Miss Under Sampling
# Train the balanced bagging classifier model using near miss under sampling
bbc_nm = BalancedBaggingClassifier(random_state=42, sampler=(NearMiss(version=3)))
bbc_nm_model_cv = cross_validate(bbc_nm, X_train, y_train, cv = 5, n_jobs = -1, scoring="recall")
# Check the model performance
print(f"{bbc_nm_model_cv['test_score'].mean():.3f} +/- {bbc_nm_model_cv['test_score'].std():.3f}")


###### Step 9: Balanced Bagging Classifier - SMOTE
# Train the balanced bagging classifier model using SMOTE
bbc_smote = BalancedBaggingClassifier(random_state=42, sampler=(SMOTE()))
bbc_smote_model_cv = cross_validate(bbc_smote, X_train, y_train, cv = 5, n_jobs = -1, scoring="recall")
# Check the model performance
print(f"{bbc_smote_model_cv['test_score'].mean():.3f} +/- {bbc_smote_model_cv['test_score'].std():.3f}")
###### Step 10: Use Best Model On Training Dataset
# Train the balanced random forest model
brf = BalancedRandomForestClassifier(random_state=42)
brf_model = eec.fit(X_train, y_train)
brf_prediction = brf_model.predict(X_test)
# Check the model performance
cf_matrix = confusion_matrix(y_test, brf_prediction)
print(cf_matrix)


print(classification_report(y_test, brf_prediction))
'''
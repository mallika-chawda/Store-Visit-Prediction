import pandas as pd
import numpy as np
import datetime

# Read Pred_form file and Create 3 separate dataframes for results of each duration
pred_form = pd.read_csv('Pred_form.csv')
pred_form['date'] = pd.to_datetime(pred_form['date']).dt.date
pred_form_1 = pred_form.iloc[:, :-3]
pred_form_3 = pred_form.iloc[:, :-3]
#pred_form_3['in_3'] = np.NaN
pred_form_5 = pred_form.iloc[:, :-3]
#pred_form_5['in_5'] = np.NaN

# Merge Test Preds
test_1 = pd.read_csv('pred_1.csv')[['StoreID', 'Date', 'Mac', 'pred_logit_test_rc']]
test_1 = test_1.groupby(['StoreID', 'Date', 'Mac'], as_index = False)['pred_logit_test_rc'].max()
test_1['Date'] = pd.to_datetime(test_1['Date']).dt.date
#print(test_1[test_1['pred_logit_test_rc']==1])

pred_1 = pred_form_1.merge(test_1, how = "left", left_on = ['StoreID', 'date', 'Mac'], right_on = ['StoreID', 'Date', 'Mac'])
print(pred_1[pred_1['pred_logit_test_rc']==1])
pred_1 = pred_1[['MallID', 'date', 'Mac', 'StoreID', 'pred_logit_test_rc']].rename(columns = {'pred_logit_test_rc':'in_1'})
#pred_1['in_1'] = pred_1['in_1'].apply(lambda x: x if not np.isnan(x) else 0)


test_3 = pd.read_csv('pred_3.csv')
test_3 = test_3.groupby(['StoreID', 'Date', 'Mac'], as_index = False)['pred_logit_test_rc'].max()
test_3['Date'] = pd.to_datetime(test_3['Date']).dt.date
pred_3 = pred_form_3.merge(test_3, how = "left", left_on = ['StoreID', 'date', 'Mac'], right_on = ['StoreID', 'Date', 'Mac'])
pred_3 = pred_3[['MallID', 'date', 'Mac', 'StoreID', 'pred_logit_test_rc']].rename(columns = {'pred_logit_test_rc':'in_3'})
pred_3['in_3'] = pred_3['in_3'].apply(lambda x: x if not np.isnan(x) else 0)
print(pred_3.head())

test_5 = pd.read_csv('pred_5.csv')
test_5 = test_5.groupby(['StoreID', 'Date', 'Mac'], as_index = False)['pred_logit_test_rc'].max()
test_5['Date'] = pd.to_datetime(test_5['Date']).dt.date
pred_5 = pred_form_5.merge(test_5, how = "left", left_on = ['StoreID', 'date', 'Mac'], right_on = ['StoreID', 'Date', 'Mac'])
pred_5 = pred_5[['MallID', 'date', 'Mac', 'StoreID', 'pred_logit_test_rc']].rename(columns = {'pred_logit_test_rc':'in_5'})
pred_5['in_5'] = pred_5['in_5'].apply(lambda x: x if not np.isnan(x) else 0)
print(pred_5.head())

pred_final = pred_1.merge(pred_3, how = 'inner', on = ['MallID', 'StoreID', 'date', 'Mac']).merge(pred_5,  how = 'inner', on = ['MallID', 'StoreID', 'date', 'Mac'])
print(pred_final.shape[0])
pred_final.to_csv('pred_form_final.csv')

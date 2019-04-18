import numpy as np
import pandas as pd
from datetime import datetime

def create_train_data(mall_id):

    # For example, for mall 2, mall_id = Mall2

    df= pd.read_csv(mall_id + "_Data.csv")

    df_r= pd.read_csv(mall_id +"_routing.csv")

    # Modifying RSSI data

    df['Time stamp'] = pd.to_datetime(df['Time stamp'])

    df['Date']= df['Time stamp'].dt.date

    df['Time']= df['Time stamp'].dt.time

    df['Hour']= df['Time stamp'].dt.hour

    df['Minute'] = df['Time stamp'].dt.minute

    df['Day'] = df['Time stamp'].dt.day

    merged_df = df.groupby(['StoreID', 'Mac', 'Date', 'Hour'], as_index=False)['RSSI'].agg(['max','min','mean','count']).reset_index()

    #grouped_df_average= df.groupby(['StoreID', 'Mac', 'Date', 'Hour'], as_index=False)['RSSI'].max()

    #grouped_df_count= df.groupby(['StoreID', 'Mac', 'Date', 'Hour'], as_index=False)['RSSI'].count()

    #merged_df = pd.merge(grouped_df_average, grouped_df_count,  how='inner', left_on=['Mac','Date', 'StoreID', 'Hour'], right_on = ['Mac','Date', 'StoreID', 'Hour'])

    merged_df= merged_df.sort_values(['Mac', 'Date', 'Hour', 'StoreID'], ascending=[True, True, True, True])

    #print(merged_df.columns)

    merged_df['One_store_ahead']= merged_df.groupby(['Mac', 'Date', 'Hour'])['mean'].shift(-1)

    merged_df['One_store_behind']= merged_df.groupby(['Mac', 'Date', 'Hour'])['mean'].shift(1)

    merged_df['One_store_ahead_count']= merged_df.groupby(['Mac', 'Date', 'Hour'])['max'].shift(-1)

    merged_df['One_store_behind_count']= merged_df.groupby(['Mac', 'Date', 'Hour'])['max'].shift(1)

    merged_df[['One_store_ahead', 'One_store_behind', 'One_store_ahead_count', 'One_store_behind_count']] = merged_df[['One_store_ahead', 'One_store_behind', 'One_store_ahead_count', 'One_store_behind_count']].fillna(value=0)

    #print(merged_df.head(402))

    # Merge with OS data
    os_data = pd.read_csv('Mac and OS.csv')
    os_data['Mac'] = os_data['Mac'].apply(lambda x: x.lower())

    merged_df = merged_df.merge(os_data, how = "left", on = "Mac")

    # print(df_train_os[df_train_os['OS'].isnull()])

    merged_df['os_number'] = merged_df['OS'].apply(lambda x: 1 if 'ios' in x.lower() else 0)

    #print(merged_df.head())

    # Modifying routing data

    df_r['DateTime'] = pd.to_datetime(df_r['Date'])

    df_r['Date_r']= df_r['DateTime'].dt.date

    df_r['Mac']= df_r['Mac'].str.lower()

    df_r['InTime']= pd.to_datetime(df_r['InTime'])

    df_r['OutTime']= pd.to_datetime(df_r['OutTime'])

    df_r['InTime_hour']= df_r['InTime'].dt.hour

    df_r['InTime_Minute'] = df_r['InTime'].dt.minute

    df_r['OutTime_hour']= df_r['OutTime'].dt.hour

    df_r['OutTime_Minute'] = df_r['OutTime'].dt.minute

    df_r['Store_number'] = df_r['StoreID'].str[-3:]

    df_r['Store_number'] = df_r['Store_number'].astype(np.uint8)

    df_r['Duration'] = pd.to_datetime(df_r['OutTime']- df_r['InTime']).dt.minute


    df_r_sub = df_r[['MallID', 'StoreID', 'Mac', 'Duration',
           'Date_r', 'InTime_hour', 'InTime_Minute', 'OutTime_hour',
           'OutTime_Minute', 'Store_number']]
           
    # Split data into durations 5, 3, and 1

    df_r_5 = df_r_sub[df_r_sub['Duration'] == 5]

    df_r_3 = df_r_sub[df_r_sub['Duration'] == 3]

    df_r_1 = df_r_sub[df_r_sub['Duration'] == 1]

    # Split into Train and Test

    # Get Date Set from Routing 

    dates_r = set(df_r['Date_r'])
    dates_probe = set(merged_df['Date'])
    reqd_dates = list(dates_probe & dates_r)

    #merged_df_train = merged_df[merged_df['Date'] >= datetime.date(datetime.strptime('2018-12-18', '%Y-%m-%d'))][merged_df['Date'] <= datetime.date(datetime.strptime('2018-12-23', '%Y-%m-%d'))]
    merged_df_train = merged_df[merged_df['Date'].isin(reqd_dates)]
    merged_df_test = pd.concat([merged_df, merged_df_train, merged_df_train]).drop_duplicates(keep=False)

    #creating data for in_1
    merge_df_1 = pd.merge(merged_df_train, df_r_1,  how='left', left_on=['StoreID', 'Mac', 'Date', 'Hour'], right_on = ['StoreID', 'Mac', 'Date_r', 'InTime_hour'])

    merge_df_1['Label'] = merge_df_1['Duration'].apply(lambda x: 1 if not np.isnan(x) else 0)
    #merge_df_1.to_csv("merged_df_1.csv")

    merge_df_1_positives = merge_df_1[merge_df_1['Label']==1]
    merge_df_1_negatives = merge_df_1[merge_df_1['Label']==0]

    merge_df_1_negatives = merge_df_1_negatives.sample(merge_df_1_positives.shape[0])

    df_train = pd.concat([merge_df_1_positives, merge_df_1_negatives])[['Mac', 'OS', 'os_number','Date', 'Hour', 'mean', 'max', 'min', 'count', 'One_store_ahead',
           'One_store_behind', 'One_store_ahead_count', 'One_store_behind_count', 'Label']]

    df_train.to_csv("train_1.csv")
    merged_df_test.to_csv("test.csv")

    #creating data for in_3
    merge_df_3 = pd.merge(merged_df_train, df_r_3,  how='left', left_on=['StoreID', 'Mac', 'Date', 'Hour'], right_on = ['StoreID', 'Mac', 'Date_r', 'InTime_hour'])

    merge_df_3['Label'] = merge_df_3['Duration'].apply(lambda x: 1 if not np.isnan(x) else 0)
    #merge_df_3.to_csv("merged_df_3.csv")

    merge_df_3_positives = merge_df_3[merge_df_3['Label']==1]
    merge_df_3_negatives = merge_df_3[merge_df_3['Label']==0]

    merge_df_3_negatives = merge_df_3_negatives.sample(merge_df_3_positives.shape[0])

    df_train = pd.concat([merge_df_3_positives, merge_df_3_negatives])[['Mac', 'OS', 'os_number', 'Date', 'Hour', 'mean', 'max', 'min', 'count', 'One_store_ahead',
           'One_store_behind', 'One_store_ahead_count', 'One_store_behind_count', 'Label']]

    df_train.to_csv("train_3.csv")

    #creating data for in_5
    merge_df_5 = pd.merge(merged_df_train, df_r_5,  how='left', left_on=['StoreID', 'Mac', 'Date', 'Hour'], right_on = ['StoreID', 'Mac', 'Date_r', 'InTime_hour'])

    merge_df_5['Label'] = merge_df_5['Duration'].apply(lambda x: 1 if not np.isnan(x) else 0)
    #merge_df_5.to_csv("merged_df_5.csv")

    merge_df_5_positives = merge_df_5[merge_df_5['Label']==1]
    merge_df_5_negatives = merge_df_5[merge_df_5['Label']==0]

    merge_df_5_negatives = merge_df_5_negatives.sample(merge_df_5_positives.shape[0])

    df_train = pd.concat([merge_df_5_positives, merge_df_5_negatives])[['Mac', 'OS', 'os_number', 'Date', 'Hour', 'mean', 'max', 'min', 'count', 'One_store_ahead', 'One_store_behind', 'One_store_ahead_count', 'One_store_behind_count', 'Label']]

    

    #print(df_train_os[df_train_os['os_number'] == 0])

    df_train.to_csv("train_5.csv")

    print(df_train.columns)



create_train_data('Mall1')
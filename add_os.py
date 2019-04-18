import pandas as pd

# Merge with OS data

merged_df = pd.read_csv('combined_csv_test.csv')

os_data = pd.read_csv('Mac and OS.csv')

os_data['Mac'] = os_data['Mac'].apply(lambda x: x.lower())

merged_df = merged_df.merge(os_data, how = "left", on = "Mac")

# print(df_train_os[df_train_os['OS'].isnull()])

merged_df['os_number'] = merged_df['OS'].apply(lambda x: 1 if 'ios' in x.lower() else 0)

merged_df.to_csv('combined_csv_test_w_os.csv')
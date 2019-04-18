import pandas as pd

malls=[1,2,3,4,5,6,7, 10, 12, 13]

filenames=[]

for mall in malls:
    filenames.append("Mall 0"+str(mall)+"/train_5.csv")

combined_csv = pd.concat([pd.read_csv(f) for f in filenames])

combined_csv.to_csv("combined_csv_train_5.csv", index=False)
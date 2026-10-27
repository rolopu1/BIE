import pandas as pd

path = "C:\\BSICoS\\Maxim\\Ensayos\\Ensayo_Oct27_1157_AddFinalBC34"

Data = pd.read_parquet(path+"\\MAX103_PPG.parquet")

Data.to_csv(path + "\\MAX103_PPG.csv", index=False)

print(Data.head())
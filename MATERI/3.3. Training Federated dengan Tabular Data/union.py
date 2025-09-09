import pandas as pd

#Memanggil Dataset
data1 = pd.read_csv('dinsos_100.csv')
data2 = pd.read_csv('dukcapil_100.csv')
data3 = pd.read_csv('kemenkes_100.csv')


#Merge
frames = [data1, data2, data3]
result = pd.concat(frames).drop_duplicates().reset_index(drop=True)

#ekspor ke new file csv
result.to_csv(r'gabungan.csv', index=False)


import pandas as pd

df = pd.read_csv('data_dummy.csv')
# print(df.to_string())

#Jika Anda memiliki DataFrame besar dengan banyak baris, 
# Pandas hanya akan mengembalikan
#  5 baris pertama dan 5 baris terakhir:
print("")
print(df)



# df = pd.read_csv('dinsos_100.csv')
# df2 = pd.read_csv('dukcapil_100.csv')
# df3 = pd.read_csv('kemenkes_100.csv')

# print("DINSOS")
# print(df)
# print("")
# print("DUKCAPIL")
# print(df2)
# print("")
# print("KEMENKES")
# print(df3)
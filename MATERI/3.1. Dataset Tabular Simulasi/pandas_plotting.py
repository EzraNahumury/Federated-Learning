import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("angka_plotting.csv")
# df.plot()
# plt.show()


# df = pd.read_csv("angka_plotting.csv")
# df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
# plt.show()

df = pd.read_csv("angka_plotting.csv")
df["Duration"].plot(kind= 'hist')
plt.show()



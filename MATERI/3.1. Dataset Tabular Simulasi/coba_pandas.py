import pandas as pd

mydataset = {
    'cars' : ["bmw", "volvo", "ford"],
    'passing' : [3,7,2]
}

myvar = pd.DataFrame(mydataset)
print(myvar)
print(pd.__version__)
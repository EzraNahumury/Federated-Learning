import pandas as pd

a = [1,7,2]
myvar = pd.Series(a)
print(myvar)
print("")

#label
print("label = ",myvar[0])
print("")


#create label
myvar = pd.Series(a,index=["x","y","z"])
print("create label : ")
print(myvar)
print("")

#retrun value y
print("retrun value y : ", myvar["y"])
print("")

#key/value objects as series
calories = {"day1": 420, "day2":220, "day3":180}
myvar = pd.Series(calories)
print(myvar)
print("")


#only frame day 1 and day 2
myvar = pd.Series(calories, index=["day1", "day2"])
print(myvar)



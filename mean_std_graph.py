import matplotlib.pyplot as plt
import os
import csv
import numpy as np
x_list = []
y_list = []
x, y = [], []
with open('mean_std.csv', 'r') as f:
    rdr = csv.reader(f)
    mylist = list(rdr)
    # print(mylist[1])
    # print(len(mylist))
    for i in range(0, len(mylist)):
        if (i % 2) == 0:
            x.append(mylist[i])
        else:
            y.append(mylist[i])
    for t in x:
        t = float(t[0])
        x_list.append(t)
    for i in y:
        i = float(i[0])
        y_list.append(i)
    print(x_list, y_list)

x_mean = np.mean(x_list)
x_std = np.std(x_list)
y_mean = np.mean(y_list)
y_std = np.std(y_list)
print(x_mean)
print(y_mean)
for i in range(0, len(x_list)): # Regularization
    x_list[i] = (x_list[i] - x_mean)/x_std

for t in range(0, len(y_list)):
    y_list[t] = (y_list[t] - y_mean)/y_std

print(x_list)
print(y_list)

plt.scatter(x_list, y_list, c="black") # 산포도 그려서 저장
plt.title("Mean, Std Distribution")
plt.xlabel("mean")
plt.ylabel("std")
plt.savefig("mean_std.png")
plt.show()
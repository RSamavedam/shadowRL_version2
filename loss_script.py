import matplotlib.pyplot as plt 
import numpy as np 
inputFile = open("best_model_mse.txt", "r")
stringList = inputFile.readlines()
numbered_list = []
for entry in stringList:
    numbered_list.append(int(float(entry[0:len(entry)-2])))

final_list = []
for i in range(0, len(numbered_list), 2):
    final_list.append((numbered_list[i] + numbered_list[i+1])/2)

x_list = []
for i in range(len(final_list)):
    x_list.append(i)

plt.plot(x_list, final_list)
plt.show()

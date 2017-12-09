import matplotlib.pyplot as plt
import sys
import os
from pylab import *

test = open("results2.txt",'r')
X=[]
y=[]
i=0
for line in test:
	s = float(line)*100
	y.append(s)
	X.append(i)
	i=i+1

plt.title("Random Forest Classifier Test Results")
plt.ylabel("Accuracy Percentage %")
plt.xlabel("Data sets")
plt.xlim([0,120])
plt.ylim([0,120])
plt.plot(X,y)
plt.savefig("images/Random_Forest_Classifier_results.jpg")
plt.clf()


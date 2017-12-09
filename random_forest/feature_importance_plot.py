import os
import sys
import pywt
from pywt import wavedec
from __init__ import ap_entropy, samp_entropy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mnist_new1 import mlp, create_training_set
from pylab import *
A=[]
B=[]
C=[]
D=[]
E=[]
for fl in os.listdir("../../../ALL/A/"):
                inp = []
                path = "../../../ALL/A/" + fl
                txt = open(path,'r')
                for line in txt:
                        feature = line.split()[0]
                        inp.append(feature)
		a = np.array(inp)
                A.append(a)

for fl in os.listdir("../../../ALL/B/"):
                inp = []
                path = "../../../ALL/B/" + fl
                txt = open(path,'r')
                for line in txt:
                        feature = line.split()[0]
                        inp.append(feature)
                a = np.array(inp)
                B.append(a)

for fl in os.listdir("../../../ALL/C/"):
                inp = []
                path = "../../../ALL/C/" + fl
                txt = open(path,'r')
                for line in txt:
                        feature = line.split()[0]
                        inp.append(feature)
                a = np.array(inp)
                C.append(a)

for fl in os.listdir("../../../ALL/D/"):
                inp = []
                path = "../../../ALL/D/" + fl
                txt = open(path,'r')
                for line in txt:
                        feature = line.split()[0]
                        inp.append(feature)
                a = np.array(inp)
                D.append(a)

for fl in os.listdir("../../../ALL/E/"):
                inp = []
                path = "../../../ALL/E/" + fl
                txt = open(path,'r')
                for line in txt:
                        feature = line.split()[0]
                        inp.append(feature)
                a = np.array(inp)
                E.append(a)

A_ = []
B_ = []
C_ = []
D_ = []
E_ = []
for x in A:
	coeffs = wavedec(x,'db4',level=8)
	A_.append(coeffs)
	
for x in B:
	coeffs = wavedec(x,'db4',level=8)
	B_.append(coeffs)
	
for x in C:
	coeffs = wavedec(x,'db4',level=8)
	C_.append(coeffs)
for x in D:
	coeffs = wavedec(x,'db4',level=8)
	D_.append(coeffs)
for x in E:
	coeffs = wavedec(x,'db4',level=8)
	E_.append(coeffs)

a=[]
b=[]
c=[]
d=[]
e=[]
y_a = []
y_b = []
y_c = []
y_d = []
y_e = []
f=[]
y_f=[]
minm = [1000000000000 for i in range(1,28)]
maxm = [0 for i in range(1,28)]
inp = []
out = []
for x in A_:
	features = []
	j=0
	for y in x:
		coef = np.array(y)
		energy = np.sum(coef**2)
		minm[j] = min(minm[j],energy)
		maxm[j] = max(maxm[j],energy)
		j=j+1
		#approx_en = ap_entropy(coef,2,0.5)
		#minm[j] = min(minm[j],approx_en)
		#maxm[j] = max(maxm[j],approx_en)
		#j=j+1
		#samp_en = samp_entropy(coef,2,0.5)
		#minm[j] = min(minm[j],samp_en)
		#maxm[j] = max(maxm[j],samp_en)
		#j=j+1
		mean = np.mean(coef)
		minm[j]= min(minm[j],mean)
		maxm[j] = max(maxm[j],mean)
		j=j+1
		std = np.std(coef)
		minm[j] = min(minm[j],std)
		maxm[j] = max(maxm[j],std)
		j=j+1
		features.append(energy)
		#features.append(approx_en)
		#features.append(samp_en)
		features.append(mean)
		features.append(std)
	a.append(features)
	y_a.append(0)

print("A done")

for x in B_:
        features = []
	j=0
        for y in x:
                coef = np.array(y)
                energy = np.sum(coef**2)
		minm[j] = min(minm[j],energy)
		maxm[j] = max(maxm[j],energy)
                j=j+1
                #approx_en = ap_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],approx_en)
		#maxm[j] = max(maxm[j],approx_en)
                #j=j+1
                #samp_en = samp_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],samp_en)
		#maxm[j] = max(maxm[j],samp_en)
                #j=j+1
                mean = np.mean(coef)
                minm[j]= min(minm[j],mean)
		maxm[j] = max(maxm[j],mean)
                j=j+1
                std = np.std(coef)
                minm[j] = min(minm[j],std)
		maxm[j]= max(maxm[j],std)
                j=j+1

                features.append(energy)
                #features.append(approx_en)
                #features.append(samp_en)
                features.append(mean)
                features.append(std)
        b.append(features)
	y_b.append(0)

print("B done")
for x in C_:
        features = []
	j=0
        for y in x:
                coef = np.array(y)
                energy = np.sum(coef**2)
		minm[j] = min(minm[j],energy)
                maxm[j] = max(maxm[j],energy)
                j=j+1
                #approx_en = ap_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],approx_en)
                #maxm[j] = max(maxm[j],approx_en)
                #j=j+1
                #samp_en = samp_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],samp_en)
                #maxm[j] = max(maxm[j],samp_en)
                #j=j+1
                mean = np.mean(coef)
                minm[j]= min(minm[j],mean)
                maxm[j] = max(maxm[j],mean)
                j=j+1
                std = np.std(coef)
                minm[j] = min(minm[j],std)
                maxm[j]= max(maxm[j],std)
                j=j+1


                features.append(energy)
                #features.append(approx_en)
                #features.append(samp_en)
                features.append(mean)
                features.append(std)
        c.append(features)
	y_c.append(1)

print("C done")
for x in D_:
        features = []
	j=0
        for y in x:
                coef = np.array(y)
                energy = np.sum(coef**2)
		
		minm[j] = min(minm[j],energy)
                maxm[j] = max(maxm[j],energy)
                j=j+1
                #approx_en = ap_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],approx_en)
                #maxm[j] = max(maxm[j],approx_en)
                #j=j+1
                #samp_en = samp_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],samp_en)
                #maxm[j] = max(maxm[j],samp_en)
                #j=j+1
                mean = np.mean(coef)
                minm[j]= min(minm[j],mean)
                maxm[j] = max(maxm[j],mean)
                j=j+1
                std = np.std(coef)
                minm[j] = min(minm[j],std)
                maxm[j]= max(maxm[j],std)
                j=j+1


                features.append(energy)
                #features.append(approx_en)
                #features.append(samp_en)
                features.append(mean)
                features.append(std)
        d.append(features)
	y_d.append(1)

print("D Done")

for x in E_:
        features = []
	j=0
        for y in x:
                coef = np.array(y)
                energy = np.sum(coef**2)
			
		minm[j] = min(minm[j],energy)
                maxm[j] = max(maxm[j],energy)
                j=j+1
                #approx_en = ap_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],approx_en)
                #maxm[j] = max(maxm[j],approx_en)
                #j=j+1
                #samp_en = samp_entropy(coef,2,0.5)
                #minm[j] = min(minm[j],samp_en)
                #maxm[j] = max(maxm[j],samp_en)
                #j=j+1
                mean = np.mean(coef)
                minm[j]= min(minm[j],mean)
                maxm[j] = max(maxm[j],mean)
                j=j+1
                std = np.std(coef)
                minm[j] = min(minm[j],std)
                maxm[j]= max(maxm[j],std)
                j=j+1



                features.append(energy)
                #features.append(approx_en)
                #features.append(samp_en)
                features.append(mean)
                features.append(std)
        e.append(features)
	f.append(features)
	y_e.append(2)
	y_f.append(2)
print("E done")
i=0
while i < 100:
	inp.append(a[i])
	out.append(y_a[i])
	inp.append(b[i])
	out.append(y_b[i])
	inp.append(c[i])
	out.append(y_c[i])
	inp.append(d[i])
	out.append(y_d[i])
	inp.append(e[i])
	out.append(y_e[i])
	inp.append(f[i])
	out.append(y_f[i])
	i=i+1

print("done")
i=0
z= [i for i in range(1,len(e)+1)]
import matplotlib.pyplot as plt
i=0
while i < 27:
        j=0
        p=[]
        q=[]
        r=[]
        s=[]
        t=[]
        u=[]
        while j < len(e):
                p.append(a[j][i])
                #q.append(b[j][i])
                r.append(c[j][i])
                #s.append(d[j][i])
                t.append(e[j][i])
                j=j+1
        plt.plot(z,np.array(p), label="normal")
        #plt.plot(z,np.array(q))
        plt.plot(z,np.array(r), label="ephilepsy without seizures")
        #plt.plot(z,np.array(s))
        plt.plot(z,np.array(t), label="ephilepsy with seizures")
	legend()
        plt.savefig("images/feature_"+str(i)+"_out.jpg")
	print("saved")
        plt.clf()
        i=i+1

from sklearn.ensemble import ExtraTreesClassifier
i=0
for x in inp:
    j=0
    while j < 27:
        den = maxm[j]-minm[j]
       	if(maxm[j]-minm[j]==0):
        	den=np.amax(np.array(maxm))
        inp[i][j] = np.float32((inp[i][j]-minm[j]))/np.float32(den)
        if(inp[i][j]<0.00001):
        	inp[i][j]=0.0001
       	if(inp[i][j]>0.99999):
               	inp[i][j]=0.9999
        inp[i][j] = np.float32(inp[i][j])
        j=j+1
    i=i+1
inp = np.array(inp)
out = np.array(out)
forest = ExtraTreesClassifier(n_estimators=100,random_state=0)
forest.fit(inp,out)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(inp.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(inp.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(inp.shape[1]), indices)
plt.xlim([-1, inp.shape[1]])
plt.savefig("images/feature_importance_out.jpg")
plt.clf()


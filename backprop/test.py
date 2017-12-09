import os
import sys
import pywt
from pywt import wavedec
from __init__ import ap_entropy, samp_entropy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mnist_new import mlp, create_training_set
from neuralnet import NeuralNet
from tools import Instance
from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function
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
                q.append(b[j][i])
                r.append(c[j][i])
                s.append(d[j][i])
                t.append(e[j][i])
                j=j+1
        plt.plot(z,np.array(p))
        plt.plot(z,np.array(q))
        plt.plot(z,np.array(r))
        plt.plot(z,np.array(s))
        plt.plot(z,np.array(t))
        plt.savefig("images/"+str(i)+"_out.jpg")
	print("saved")
        plt.clf()
        i=i+1

settings = {
    # Required settings
    "n_inputs"              : 27,       # Number of network input signals
    "layers"                : [ (300, sigmoid_function), (200, sigmoid_function), (1, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in you list describes the number of output signals
    
    # Optional settings
    "weights_low"           : 0.001,     # Lower bound on initial weight range
    "weights_high"          : 0.01,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.0,      # dropout fraction in all hidden layers
}

training_set=[]

i=0
for x in inp:
    j=0
    while j < 27:
	den = maxm[j]-minm[j]
	if(maxm[j]-minm[j]==0):
		den=10000
        x[j] = np.float32((x[j]-minm[j]))/np.float32(den)
	if(x[j]<0.00001):
		x[j]=0.0001
	if(x[j]>0.99999):
		x[j]=0.9999
        x[j] = np.float32(x[j])
        j=j+1
    training_set.append(Instance(x,[out[i]/2.0]))
    i=i+1
import ipdb;
ipdb.set_trace()
training_one = training_set

network = NeuralNet( settings )

#network.scg(
#                training_one, 
#                ERROR_LIMIT = 1e-4
#            )
network.backpropagation( 
                training_one,          # specify the training set
                ERROR_LIMIT     = 1e-3, # define an acceptable error limit 
                #max_iterations  = 100, # continues until the error limit is reach if this argument is skipped

                # optional parameters
                learning_rate   = 0.001, # learning rate
                momentum_factor = 0.9, # momentum
            )

print "Final MSE:", network.test( training_one )


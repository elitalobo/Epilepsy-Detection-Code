from pylab import *
import matplotlib.pyplot as plt
test = open("final_results1.txt",'r')
X = []
y = []
i=0
for line in test:	
	s = line.split(' ')
	s1 = (s[3].split('\t'))
	try:
		y.append(float(s1[2]))
		X.append(i)
	except:
		print(s1)
	i=i+1
print(y)
plt.title("Convolution Network Learning Curve")
plt.plot(X,np.array(y))
plt.savefig("images/convolution_network_results_2.jpg")
print("saved")
plt.clf()


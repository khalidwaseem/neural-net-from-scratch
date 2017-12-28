import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


################################
# Each column is one datapoint.
################################



###################################################################
# Function to visualize generated dataset (X : points, Y : label)
###################################################################
def visualize_dataset(X, Y):
	plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
	plt.show()



###################################################################
# Function to generate flower dataset (X : points, Y : label)
###################################################################
def generate_flower_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    K = 2 # number of classes
    N = int(m/K) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    # Visualize the data:
    #visualize_dataset(X,Y)

    return X, Y



###################################################################
# Function to generate spiral dataset (X : points, Y : label)
###################################################################
def generate_spiral_dataset():
	np.random.seed(1)
	m = 400 # number of examples
	K = 2 # number of classes
	N = int(m/K) # number of points per class
	D = 2 # dimensionality
	X = np.zeros((m,D)) # data matrix (each row = single example)
	Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
	
	for j in range(K):
  		ix = range(N*j,N*(j+1))
  		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  		r = np.linspace(0.0,1,N) # radius
  		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  		Y[ix] = j

  	X = X.T
	Y = Y.T

	# Visualize the data:
	#visualize_dataset(X,Y)

	return X, Y



###################################################################
# Function to generate noisy circles dataset (X : points, Y : label)
###################################################################
def generate_noisy_circles_dataset():
	N = 400
	
	X, Y = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
	X, Y = X.T, Y.reshape(1, Y.shape[0])

	# Visualize the data:
	#visualize_dataset(X,Y)

	return X, Y



###################################################################
# Function to generate noisy moons dataset (X : points, Y : label)
###################################################################
def generate_noisy_moons_dataset():
	N = 400
	
	X, Y = sklearn.datasets.make_moons(n_samples=N, noise=.2)
	X, Y = X.T, Y.reshape(1, Y.shape[0])

	# Visualize the data:
	#visualize_dataset(X,Y)

	return X, Y



###########################################################################
# Function to generate gaussian quantiles dataset (X : points, Y : label)
###########################################################################
def generate_gaussian_quantiles_dataset():
	N = 400
	
	X, Y = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
	X, Y = X.T, Y.reshape(1, Y.shape[0])

	# Visualize the data:
	#visualize_dataset(X,Y)

	return X, Y



##############################################################
# Function to generate blobs dataset (X : points, Y : label)
##############################################################
def generate_blobs_dataset():
	N = 400
	
	X, Y = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
	X, Y = X.T, Y.reshape(1, Y.shape[0])
	Y = Y%2

	# Visualize the data:
	#visualize_dataset(X,Y)

	return X, Y



###########################################################################
# Function to generate dataset (X : points, Y : label)
# Supported datasets are : flower, spiral, noisy_circles, noisy_moons, 
# gaussian_quantiles, blobs
###########################################################################
def generate_dataset(dataset_name):
	X, Y = [],[]
	
	if dataset_name == 'flower':
		X, Y = generate_flower_dataset()
	elif dataset_name == 'spiral':
		X, Y = generate_spiral_dataset()
	elif dataset_name == 'noisy_circles':
		X, Y = generate_noisy_circles_dataset()
	elif dataset_name == 'noisy_moons':
		X, Y = generate_noisy_moons_dataset()
	elif dataset_name == 'gaussian_quantiles':
		X, Y = generate_gaussian_quantiles_dataset()
	elif dataset_name == 'blobs':
		X, Y = generate_blobs_dataset()

	return X, Y



#generate_dataset('flower')
#generate_dataset('spiral')
#generate_dataset('noisy_circles')
#generate_dataset('noisy_moons')
#generate_dataset('gaussian_quantiles')
#generate_dataset('blobs')

			

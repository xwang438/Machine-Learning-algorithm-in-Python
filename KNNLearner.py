import numpy as np

class KNNLearner:
	"""
    Implement and evaluate a KNN learner class. 
    Implement the following functions/methods for the KNN learner:
    learner = KNNLearner(k = 3)
    learner.addEvidence(Xtrain, Ytrain)
    Y = learner.query(Xtest)
    Where "k" is the number of nearest neighbors to find. 
    Xtrain and Xtest are ndarrays (numpy objects) where each row represents an X1, X2, X3... XN set of feature values. 
    The columns are the features and the rows are the individual example instances. 
    Y and Ytrain are single dimensional lists that indicate the value we are attempting to predict with X. 
    """
	def __init__(self, k):
		self.k = k

	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain

	def query(self, Xtest):

		classify = np.zeros(len(Xtest))
		
		# For each element in test data, compute the distance with each element in train data
		# Find the k nearest elements in train data
		
		for j in range(0, len(Xtest)):
			distance = np.zeros(len(self.Xtrain))
			
			for i in range(0, len(self.Xtrain)):
				for l in range(0, Xtest.shape[1]):
					distance[i] += (self.Xtrain[i, l] - Xtest[j, l]) * (self.Xtrain[i, l] - Xtest[j, l]);
					
			indices = np.argsort(distance)
			selectn = np.zeros(self.k)
			for n in range(0, self.k):
			    selectn[n] = self.Ytrain[indices[n]]
			classify[j] = np.mean(selectn)
			
		return classify
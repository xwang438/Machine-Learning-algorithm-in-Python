import numpy as np

class LinRegLearner:
	"""
    Implement and evaluate a Linear Regression learner class. 
    Implement the following functions/methods for the Linear Regression learner:
    learner = LinRegLearner()
    learner.addEvidence(Xtrain, Ytrain)
    Y = learner.query(Xtest)
    Where "k" is the number of nearest neighbors to find. 
    Xtrain and Xtest are ndarrays (numpy objects) where each row represents an X1, X2, X3... XN set of feature values. 
    The columns are the features and the rows are the individual example instances. 
    Y and Ytrain are single dimensional lists that indicate the value we are attempting to predict with X. 
    """
	def __init__(self):
		self.Xtrain = None
		self.Ytrain = None
		self.coeff = None
		self.res = None

	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain

		matrix = np.hstack([self.Xtrain, np.ones((len(self.Xtrain[:, 0]), 1))])
		self.coeff = np.zeros(2)
		self.coeff[0] = np.linalg.lstsq(matrix, Ytrain)[0][0]
		self.coeff[1] = np.linalg.lstsq(matrix, Ytrain)[0][1]
		self.res = np.linalg.lstsq(matrix, Ytrain)[0][2]

	def query(self, Xtest):
		Y = np.dot(Xtest, self.coeff) + self.res
		return Y
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import csv
import KNNLearner as knn
import LinRegLearner as lrl
import time
import math

"""
evaluates KNNLearner and LinRegLearner in the following manner:
Selects the first 60 percent of the data for training (e.g., feed to addEvidence().
Use the remaining 40 percent for testing (e.g., query).
Evaluate the following for each learner:
Time required for training (average seconds per instance)
Time required for query (average seconds per instance)
RMS error (root mean square difference between "correct" Y and the result provided by query()). See http://en.wikipedia.org/wiki/Root_mean_square for example of how to calculate RMS error.
Correlation coefficient of the response from the learner versus the correct response (using the 40 persent out of sample data)
"""

def testlearner(filename, learner):
	#generate the train set, test set
	#count train time, query time
	#compute RMSE, correlation coefficient
	reader = csv.reader(open(filename, 'rU'), delimiter=',')
	totalrows = len(open(filename).readlines())
	trainrows = totalrows * 0.6
	Xtrain = np.ndarray(shape=(trainrows, 2))
	Ytrain = np.zeros(trainrows)
	Xtest = np.ndarray(shape=(totalrows - trainrows, 2))
	Ytest = np.zeros(totalrows - trainrows)
	count = 0
	for row in reader:
		if count < trainrows:
			Xtrain[count][0] = float(row[0])
			Xtrain[count][1] = float(row[1])
			Ytrain[count] = float(row[2])
		elif count >= trainrows:
			Xtest[count - trainrows][0] = float(row[0])
			Xtest[count - trainrows][1] = float(row[1])
			Ytest[count - trainrows] = float(row[2])
		count += 1

	start1 = time.clock()
	learner.addEvidence(Xtrain, Ytrain)
	stop1 = time.clock()
	traintime = (stop1 - start1) / len(Xtrain)
	start2 = time.clock()
	Y = learner.query(Xtest)
	stop2 = time.clock()
	querytime = (stop2 - start2) / len(Xtrain)
	rms = np.linalg.norm(Y - Ytest) / np.sqrt(len(Y))
	a = len(Y) * np.dot(Y, Ytest)
	b = np.sum(Ytest) * np.sum(Y)
	c = len(Y) * math.pow(np.linalg.norm(Ytest), 2) - math.pow(np.sum(Ytest), 2)
	d = len(Y) * math.pow(np.linalg.norm(Y), 2) - math.pow(np.sum(Y), 2)
	e = math.sqrt(c * d)
	corr = (a - b) / e
	Y_in_sample = learner.query(Xtrain)
	rms_in_sample = np.linalg.norm(Y_in_sample - Ytrain) / np.sqrt(len(Y_in_sample))

	return [Xtrain, Ytrain, Xtest, Ytest, traintime, querytime, rms, corr, Y, rms_in_sample]


if __name__ == '__main__':

	# Plot the correlation coefficient for data-classification-prob.csv and data-ripple-prob.csv
    # As K varies from 1 to 100
   
	krange = 50
	correlation1 = np.zeros(krange)
	correlation2 = np.zeros(krange)
	train_time1 = np.zeros(krange)
	train_time2 = np.zeros(krange)
	query_time1 = np.zeros(krange)
	query_time2 = np.zeros(krange)
	rms1 = np.zeros(krange)
	rms2 = np.zeros(krange)
	rms_in_sample1 = np.zeros(krange)
	rms_in_sample2 = np.zeros(krange)
	rms1[0] = 1
	rms2[0] = 1

	for k in range(1, krange):
		[Xtrain, Ytrain, Xtest, Ytest, traintime, querytime, rms, corr, Y, rms_in_sample] = testlearner('data-classification-prob.csv', knn.KNNLearner(k))
		train_time1[k] = traintime
		query_time1[k] = querytime
		rms1[k] = rms
		correlation1[k] = corr
		rms_in_sample1[k] = rms_in_sample

		[Xtrain, Ytrain, Xtest, Ytest, traintime, querytime, rms, corr, Y, rms_in_sample] = testlearner('data-ripple-prob.csv', knn.KNNLearner(k))
		train_time2[k] = traintime
		query_time2[k] = querytime
		rms2[k] = rms
		correlation2[k] = corr
		rms_in_sample2[k] = rms_in_sample


	k = range(0, krange);

	plt.clf()

	subplot(3, 1, 1)
	plt.plot(k, train_time1)
	plt.title('data-classification-prob.csv')

	plt.xlabel('K')
	plt.ylabel('avg train time per instance')

	subplot(3, 1, 3)
	plt.plot(k, train_time2)
	plt.title('data-ripple-prob.csv')

	plt.xlabel('K')
	plt.ylabel('avg train time per instance')

	plt.savefig('train_time')

	plt.clf()

	subplot(3, 1, 1)
	plt.plot(k, query_time1)
	plt.title('data-classification-prob.csv')

	plt.xlabel('K')
	plt.ylabel('avg query time per instance')

	subplot(3, 1, 3)
	plt.plot(k, query_time2)
	plt.title('Tdata-ripple-prob.csv')

	plt.xlabel('K')
	plt.ylabel('avg query time per instance')

	plt.savefig('query_time')

	plt.clf()

	subplot(3, 1, 1)
	plt.plot(k, rms1)
	plt.title('data-classification-prob.csv')

	plt.xlabel('K')
	plt.ylabel('rms')

	subplot(3, 1, 3)
	plt.plot(k, rms2)
	plt.title('data-ripple-prob.csv')

	plt.xlabel('K')
	plt.ylabel('rms')

	plt.savefig('RMS')

	plt.clf()

	subplot(3, 1, 1)
	plt.plot(k, correlation1)
	plt.title('data-classification-prob.csv')

	plt.xlabel('K')
	plt.ylabel('correlation')

	subplot(3, 1, 3)
	plt.plot(k, correlation2)
	plt.title('data-ripple-prob.csv')

	plt.xlabel('K')
	plt.ylabel('correlation')

	plt.savefig('correlation')

	plt.clf()
	subplot(3, 1, 1)
	plt.plot(k, rms_in_sample1, label="in-sample error")
	plt.plot(k, rms1, label="out-of-sample error", color='r')
	plt.title('data-classification-prob.csv')

	plt.xlabel('K')
	plt.ylabel('error')

	subplot(3, 1, 3)
	plt.plot(k, rms_in_sample2, label="in-sample error")
	plt.plot(k, rms2, label="out-of-sample error", color='r')
	plt.title('data-ripple-prob.csv')

	plt.xlabel('K')
	plt.ylabel('error')

	plt.savefig('in-sample_error_versus_out-of-sample_error')


	index1 = np.argmax(correlation1)
	index2 = np.argmax(correlation2)
	index3 = np.argmin(rms1)
	index4 = np.argmin(rms2)

	print "For KNN, best correlation for file1 occurs when k is :", index1
	print "For KNN, best correlation for file2 occurs when k is :", index2
	print "For KNN, minimal rmse for file1 occurs when k is :", index3
	print "For KNN, minimal rmse for file2 occurs when k is :", index4

	[Xtrain, Ytrain, Xtest, Ytest, traintime, querytime, rms, corr, Y, rms_in_sample] = testlearner('data-classification-prob.csv', knn.KNNLearner(index1))
	[Xtrain2, Ytrain2, Xtest2, Ytest2, traintime2, querytime2, rms2, corr2, Y2, rms_in_sample2] = testlearner('data-ripple-prob.csv', knn.KNNLearner(index2))

	plt.clf()
	subplot(3, 1, 1)
	x = range(0, len(Y));
	plt.plot(x, Y, 'ro')
	plt.plot(x, Ytest, 'go')
	plt.title('data-classification-prob.csv')

	plt.xlabel('test X')
	plt.ylabel('error')

	subplot(3, 1, 3)
	plt.plot(x, Y2, 'ro')
	plt.plot(x, Ytest2, 'go')
	plt.title('data-ripple-prob.csv')

	plt.xlabel('test X')
	plt.ylabel('error')

	plt.savefig('predicted_Y_versus_actual_Y')

    
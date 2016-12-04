from multiprocessing import Pool, TimeoutError
import os
import time
import pickle
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# class myThread (threading.Thread):
#     def __init__(self, threadID, name, counter, X_val, Y_val, num_matches):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#         self.X_val =  X_val
#         self.Y_val = Y_val
#         self.num_matches = num_matches
#     def run(self):
#         print "Starting " + self.name
#         print "X_val",self.X_val
#         correct_predictions = testrun(self.X_val, self.Y_val, self.num_matches)
#         # testrun()
#         # print_time(self.name, self.counter, 5)
#         print "Exiting " + self.name


NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES*2



def my_distance(vec1,vec2):
    return np.sum(np.logical_and(vec1,vec2))

def poly_weights_evaluate(distances):
    '''Returns a list of weights given a polynomial weighting function'''
    # distances = distances[0]
    # weights = (distances * 0.1)
    # weights = weights ** 15
    weights = np.power(np.multiply(distances[0], 0.1), 4)
    return np.array([weights])

def testrun_mode(full_list):
    testrun(full_list[0],full_list[1], full_list[2])

def testrun(X_val,Y_val, num_matches):
    correct_predictions = 0
    Y_pred = np.zeros(num_matches)
    # print "Ypred",Y_pred.shape
    for i, radiant_query in enumerate(X_val):
        dire_query = np.concatenate((radiant_query[NUM_HEROES:NUM_FEATURES], radiant_query[0:NUM_HEROES]))
        rad_prob = model.predict_proba(radiant_query)[0][1]
        dire_prob = model.predict_proba(dire_query)[0][0]
        overall_prob = (rad_prob + dire_prob) / 2
        prediction = 1 if (overall_prob > 0.5) else -1
        Y_pred[i] = 1 if prediction == 1 else 0
        result = 1 if prediction == Y_val[i] else 0
        correct_predictions += result
        print "Current loop ended i-> ",i,"correct predictions",correct_predictions
        print "Time till now", (time.time() - start_time)

    # accuracy = float(correct_predictions) / NUM_MATCHES
    # print 'Accuracy of KNN model: %f' % accuracy

    # flip all -1 true labels to 0 for f1 scoring
    # for i, match in enumerate(Y):
    #     if match == -1:
    #         Y[i] = 0
    #
    # prec, recall, f1, support = precision_recall_fscore_support(Y, Y_pred, average='macro')
    # print 'Precision: ',prec
    # print 'Recall: ',recall
    # print 'F1 Score: ',f1
    # print 'Support: ',support

    # Accuracy of KNN model: 0.678074
    # Precision:  0.764119601329
    # Recall:  0.673499267936
    # F1 Score:  0.715953307393
    # Support:  3415
    # print "Time Taken in seconds ", (time.time() - start_time)

# Import the test x matrix and Y vector
print "before this?"
preprocessed = np.load('test_5669.npz')
print "after this?"
X = preprocessed['X']
Y = preprocessed['Y']
# print len(X)
X = X[0:100]
Y = Y[0:100]
# print len(X)
# print X[0:10000]
# print "X",X[0]
# print "is tihs even being considered?"
X1 = X[0:20]
X2 = X[21:40]
X3 = X[41:60]
X4 = X[61:80]
X5 = X[81:100]

Y1 = Y[0:20]
Y2 = Y[21:40]
Y3 = Y[41:60]
Y4 = Y[61:80]
Y5 = Y[81:100]

NUM_MATCHES_1 = len(X1)
NUM_MATCHES_2 = len(X2)
NUM_MATCHES_3 = len(X3)
NUM_MATCHES_4 = len(X4)
NUM_MATCHES_5 = len(X5)

NUM_MATCHES = len(X)
print "NUmMatches",NUM_MATCHES

if __name__ == '__main__':
    print "whaaa?"
    with open('evaluate_model_50000.pkl', 'r') as input_file:
            model = pickle.load(input_file)
    start_time = time.time()

    list_1 = []
    list_1.append(X1)
    list_1.append(Y1)
    list_1.append(NUM_MATCHES_1)

    list_2 = []
    list_2.append(X2)
    list_2.append(Y2)
    list_2.append(NUM_MATCHES_2)

    list_3 = []
    list_3.append(X3)
    list_3.append(Y3)
    list_3.append(NUM_MATCHES_3)

    list_4 = []
    list_4.append(X4)
    list_4.append(Y4)
    list_4.append(NUM_MATCHES_4)

    list_5 = []
    list_5.append(X5)
    list_5.append(Y5)
    list_5.append(NUM_MATCHES_5)




    total_list = []
    total_list.append(list_1)
    total_list.append(list_2)
    total_list.append(list_3)
    total_list.append(list_4)
    total_list.append(list_5)



    pool = Pool(processes=5)              # start 2 worker processes
    pool.map(testrun_mode, total_list)
    correct_predictions = 0
    # Y_pred = np.zeros(NUM_MATCHES)
    # print "Ypred",Y_pred.shape
    # thread1 = myThread(1, "Thread-1", 1, X1, Y1, NUM_MATCHES_1)
    # thread2 = myThread(2, "Thread-2", 2, X2, Y2, NUM_MATCHES_2)
    #
    # thread1.start()
    # thread2.start()
    print "Time Taken in seconds ", (time.time() - start_time)
    # test(X)

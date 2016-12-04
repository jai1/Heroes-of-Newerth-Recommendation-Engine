from multiprocessing import Pool, TimeoutError
import time
import pickle
import numpy as np



NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES*2



def my_distance(vec1,vec2):
    return np.sum(np.logical_and(vec1,vec2))

def poly_weights_evaluate(distances):
    weights = np.power(np.multiply(distances[0], 0.1), 4)
    return np.array([weights])

def run_exp_divider(full_list):
    run_exp(full_list[0],full_list[1], full_list[2])

def run_exp(X_val,Y_val, num_matches):
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


# Import the test x matrix and Y vector
print "before this?"
preprocessed = np.load('test_5669.npz')
print "after this?"
X = preprocessed['X']
Y = preprocessed['Y']
# print len(X)
# X = X[0:100]
# Y = Y[0:100]
# print len(X)
# print X[0:10000]
# print "X",X[0]
# print "is tihs even being considered?"
X1 = X[0:1000]
X2 = X[10001:2000]
X3 = X[20001:3000]
X4 = X[30001:4000]
X5 = X[40001:5000]

Y1 = Y[0:1000]
Y2 = Y[10001:2000]
Y3 = Y[20001:3000]
Y4 = Y[30001:4000]
Y5 = Y[40001:5000]

NUM_MATCHES_1 = len(X1)
NUM_MATCHES_2 = len(X2)
NUM_MATCHES_3 = len(X3)
NUM_MATCHES_4 = len(X4)
NUM_MATCHES_5 = len(X5)

NUM_MATCHES = len(X)
print "NUmMatches",NUM_MATCHES

if __name__ == '__main__':
    print "whaaa?"
    with open('evaluate_model__RF_50000.pkl', 'r') as input_file:
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
    pool.map(run_exp_divider, total_list)
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

from multiprocessing import Pool, TimeoutError
import os
import time
import pickle
from plistlib import Data
from reportlab.graphics.samples.scatter import Scatter
import matplotlib.pyplot as plt
import numpy as np

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()
# /home/akshaya/PycharmProjects/MLProject/HONRE/Heroes-of-Newerth-Recommendation-Engine/newerth/test.npz
# /home/akshaya/PycharmProjects/MLProject/HONRE/Heroes-of-Newerth-Recommendation-Engine/newerth/train.npz
NUM_HEROES = 249
NUM_FEATURES = NUM_HEROES*2
percent_val = 100
model = None

## Calculates the distance between two vectors
def my_distance(vec1,vec2):
    return np.sum(np.logical_and(vec1,vec2))

## Calculates the list of weights and returns it
def poly_weights_evaluate(distances):
    weights = np.power(np.multiply(distances[0], 0.1), 4)
    return np.array([weights])

def run_mode(full_list):
    return final_run(full_list[0],full_list[1], full_list[2])

def final_run(X_val,Y_val, num_matches):
    correct_predictions = 0
    Y_pred = np.zeros(num_matches)
    # print "Ypred",Y_pred.shape
    accuracy_list = []
    total_elements_list = []

    ## Run the loop for checking each value
    for i, radiant_query in enumerate(X_val):

        ## Calc dire query
        dire_query = np.concatenate((radiant_query[NUM_HEROES:NUM_FEATURES], radiant_query[0:NUM_HEROES]))

        ## Calc radiand and dire probability
        rad_prob = model.predict_proba(radiant_query)[0][1]
        dire_prob = model.predict_proba(dire_query)[0][0]

        ## Calc Overall Probability
        overall_prob = (rad_prob + dire_prob) / 2

        ## Prediction if overall prob > 0.5
        prediction = 1 if (overall_prob > 0.5) else -1

        ## Ypred given the value of pred
        Y_pred[i] = 1 if prediction == 1 else 0
        result = 1 if prediction == Y_val[i] else 0

        ## correct prediction increases based on result
        correct_predictions += result
        print "Current loop ended i-> ",i,"correct predictions",correct_predictions
        # print "Time till now", (time.time() - start_time)
        if (i+1)%200 ==0:
            current_accuracy = (correct_predictions/((i+1)*1.0)) * percent_val
            accuracy_list.append(current_accuracy)
            total_elements_list.append(i)
            # print "current_accuracy",current_accuracy
            # print accuracy_list,total_elements_list
    return correct_predictions

def run():
    # print "whaaa?"
    # with open('evaluate_model_50000.pkl', 'r') as input_file:
    # with open('evaluate_model_37782.pkl', 'r') as input_file:
    #         model = pickle.load(input_file)
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


    final_correct_predictions = []
    pool = Pool(processes=5)              # start 5 worker processes
    final_correct_predictions = pool.map(run_mode, total_list)

    sum_correct_prediction = 0
    for i in final_correct_predictions:
        sum_correct_prediction = sum_correct_prediction + i
    accuracy = (sum_correct_prediction/(len(X)*1.0)) * 100
    # pool.close()
    # pool.join()
    print "final correct predictions:",final_correct_predictions
    print "total correct predictions:",sum_correct_prediction
    print "accuracy:", accuracy
    print "Time Taken in seconds ", (time.time() - start_time)

    # print "acc",accuracy_list,"total_element",total_elements_list
    # plt.plot(total_elements_list, accuracy_list)
    # plt.plot([20.0, 35.0, 40.0, 50.0, 46.0, 41.66666666666667, 41.42857142857143, 38.75, 40.0, 40.0] ,
    #          [9, 19, 29, 39, 49, 59, 69, 79, 89, 99])
    plt.axis([0,4400,0,100])
    plt.show()


if __name__ == '__main__':
    with open('evaluate_model_37782.pkl', 'r') as input_file:
        model = pickle.load(input_file)
    # Import the test x matrix and Y vector
    print "before this?"
    preprocessed = np.load('test.npz')
    print "after this?"
    X = preprocessed['X']
    Y = preprocessed['Y']
    # print len(X)
    X = X[0:100]
    Y = Y[0:100]
    # print len(X)
    # print X[0:10000]
    # print "X",X[0]
    # print "is this even being considered?"
    X1 = X[0:20]
    X2 = X[20:40]
    X3 = X[40:60]
    X4 = X[60:80]
    X5 = X[80:100]

    Y1 = Y[0:20]
    Y2 = Y[20:40]
    Y3 = Y[40:60]
    Y4 = Y[60:80]
    Y5 = Y[80:100]

    # X1 = X[0:800]
    # X2 = X[800:1600]
    # X3 = X[1600:2400]
    # X4 = X[2400:3200]
    # X5 = X[3200:4000]
    #
    # Y1 = Y[0:800]
    # Y2 = Y[800:1600]
    # Y3 = Y[1600:2400]
    # Y4 = Y[2400:3200]
    # Y5 = Y[3200:4000]

    NUM_MATCHES_1 = len(X1)
    NUM_MATCHES_2 = len(X2)
    NUM_MATCHES_3 = len(X3)
    NUM_MATCHES_4 = len(X4)
    NUM_MATCHES_5 = len(X5)

    NUM_MATCHES = len(X)
    print "NUmMatches", NUM_MATCHES
    run()
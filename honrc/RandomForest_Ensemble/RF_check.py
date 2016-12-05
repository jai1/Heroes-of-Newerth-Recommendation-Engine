from multiprocessing import Pool
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

NUM_HEROES = 249
NUM_FEATURES = NUM_HEROES*2
percent_val = 100

def my_distance(vec1,vec2):
    return np.sum(np.logical_and(vec1,vec2))

def poly_weights_evaluate(distances):
    weights = np.power(np.multiply(distances[0], 0.1), 4)
    return np.array([weights])

def run_exp_divider(full_list):
    # correct_predictions,accuracy_list,total_elements_list =
    return run_exp(full_list[0],full_list[1], full_list[2])

def run_exp(X_val,Y_val, num_matches):
    correct_predictions = 0
    # count = 0 Testing for the count values
    Y_pred = np.zeros(num_matches)
    # print "Ypred",Y_pred.shape
    accuracy_list = []
    total_elements_list = []
    for i, radiant_query in enumerate(X_val):
        dire_query = np.concatenate((radiant_query[NUM_HEROES:NUM_FEATURES], radiant_query[0:NUM_HEROES]))

        ##Calculate the rad_prob & dire_Prob
        rad_prob = model.predict_proba(radiant_query)[0][1]
        dire_prob = model.predict_proba(dire_query)[0][0]

        ## calculates the overall prob
        overall_prob = (rad_prob + dire_prob) / 2

        ## prediction is true if overall prob > 0.5
        prediction = 1 if (overall_prob > 0.5) else -1

        #Y_prediction is set to 1 if prediction is 1
        Y_pred[i] = 1 if prediction == 1 else 0


        result = 1 if prediction == Y_val[i] else 0
        correct_predictions += result
        print "Current loop ended i-> ",i,"correct predictions",correct_predictions
        # print "Time till now", (time.time() - start_time)
        # count = count +1
        ## Calculating accuracy every
        if (i+1)%200 ==0:
            current_accuracy = (correct_predictions/((i+1)*1.0)) * percent_val
            accuracy_list.append(current_accuracy)
            total_elements_list.append(i)
            # print "current_accuracy",current_accuracy
            # print accuracy_list,total_elements_list
    ## Updates all three values in the list
    list_temp = []
    list_temp.append(correct_predictions)
    list_temp.append(accuracy_list)
    list_temp.append(total_elements_list)
    return list_temp


# Import the test x matrix and Y vector
preprocessed = np.load('test_5669.npz')
# preprocessed = np.load('train.npz')

X = preprocessed['X']
Y = preprocessed['Y']

# print len(X)
# print "is tihs even being considered?"

#To get absolute result till 4000
# X = X[0:40]
# Y=Y[0:40]

## Testing using multiprocess large sample
# X1 = X[0:500]
# X2 = X[500:]
# X3 = X[2300:3450]
# X4 = X[3450:4600]
# X5 = X[4600:]
#
# Y1 = Y[0:500]
# Y2 = Y[500:]
# Y3 = Y[2300:3450]
# Y4 = Y[3450:4600]
# Y5 = Y[4600:]

## Testing using multiprocess short sample
# X1 = X[0:10]
# X2 = X[10:20]
# X3 = X[20:30]
# X4 = X[30:40]
# X5 = X[40:50]
#
# Y1 = Y[0:10]
# Y2 = Y[10:20]
# Y3 = Y[20:30]
# Y4 = Y[30:40]
# Y5 = Y[40:50]


# NUM_MATCHES_1 = len(X1)
# NUM_MATCHES_2 = len(X2)
# NUM_MATCHES_3 = len(X3)
# NUM_MATCHES_4 = len(X4)
# NUM_MATCHES_5 = len(X5)

NUM_MATCHES = len(X)
print "NUmMatches",NUM_MATCHES

if __name__ == '__main__':
    ## Use the evaluate_model based on the options
    with open('evaluate_model__RF_37782.pkl', 'r') as input_file:
            model = pickle.load(input_file)
    start_time = time.time()

    ## Adding 5 different lists for each process
    # list_1 = []
    # list_1.append(X1)
    # list_1.append(Y1)
    # list_1.append(NUM_MATCHES_1)
    #
    # list_2 = []
    # list_2.append(X2)
    # list_2.append(Y2)
    # list_2.append(NUM_MATCHES_2)
    #
    # list_3 = []
    # list_3.append(X3)
    # list_3.append(Y3)
    # list_3.append(NUM_MATCHES_3)
    #
    # list_4 = []
    # list_4.append(X4)
    # list_4.append(Y4)
    # list_4.append(NUM_MATCHES_4)
    #
    # list_5 = []
    # list_5.append(X5)
    # list_5.append(Y5)
    # list_5.append(NUM_MATCHES_5)

    ## As of now using a single feature
    list_main = []
    list_main.append(X)
    list_main.append(Y)
    list_main.append(NUM_MATCHES)

    ## Contains all the list for multiprocessing
    total_list = []
    # total_list.append(list_1)
    # total_list.append(list_2)
    # total_list.append(list_3)
    # total_list.append(list_4)
    # total_list.append(list_5)
    total_list.append(list_main)

    # accuracy_list=[]
    # total_elements_list = []

    final_correct_predictions = []
    pool = Pool(processes=5)              # start 5 worker processes

    ## Currently just working with single process since Random Forest is fast
    final_correct_predictor = pool.map(run_exp_divider, total_list)
    final_correct_predictions = final_correct_predictor[0]
    accuracy_list = final_correct_predictions[1]
    total_elements_list = final_correct_predictions[2]

    # sum_correct_prediction = 0
    # for i in final_correct_predictions:
    #     sum_correct_prediction = sum_correct_prediction + i
    # accuracy = (sum_correct_prediction/(len(X)*1.0)) * 100
    # # pool.close()
    # # pool.join()
    # print "final correct predictions:",final_correct_predictions
    # print "total correct predictions:",sum_correct_prediction
    # print "accuracy:", accuracy
    print "Time Taken in seconds ", (time.time() - start_time)

    # print "acc",accuracy_list,"total_element",total_elements_list

    ## Plotting the graphs based on the data received
    plt.plot(total_elements_list, accuracy_list)
    plt.axis([0,4400,0,100])
    plt.show()
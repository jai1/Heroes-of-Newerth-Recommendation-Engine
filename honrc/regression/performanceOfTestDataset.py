import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle, os

NUM_OF_HEROES = 249
NUM_OF_FEATURES = NUM_OF_HEROES * 2

def score(model, inputVector):
    # Reverse doesn't work here
    hellbourne_first_input_vector = np.concatenate((inputVector[NUM_OF_HEROES:NUM_OF_FEATURES], inputVector[0:NUM_OF_HEROES]))
    legion_win_probability = model.predict_proba(inputVector)[0][1]
    dire_win_probability = model.predict_proba(hellbourne_first_input_vector)[0][0]
    return (legion_win_probability + dire_win_probability) / 2

def make_prediction(model, inputVector):
    prob = score(model, inputVector)
    # POSITIVE_LABEL = 1
    # NEGATIVE_LABEL = 0
    return 1 if prob > 0.5 else 0

def extractModel():
    with open("./model.pkl", 'rb') as input_file:
        u = pickle._Unpickler(input_file)
        u.encoding = 'latin1'
        return u.load()

def run():
    model = extractModel()

    testing_data = np.load('./train.npz')
    X_test = testing_data['X']
    Y_test = testing_data['Y']
    NUM_OF_MATCHES = len(Y_test)

    Y_pred = np.zeros(NUM_OF_MATCHES)
    for i, inputVector in enumerate(X_test):
        Y_pred[i] = int(make_prediction(model, inputVector))
    '''
    print("**************************************")
    print(*Y_test, sep='   ')
    print("++++++++++++++++++++++++++++++++++++++")
    print(*Y_pred , sep='   ')
    print("**************************************")
    '''
    # From Stackoverflow
    prec, recall, f1, support = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
    print("*******************Resukt***************")
    print('Precision: ',prec)
    print('Recall: ',recall)
    print('F1 Score: ',f1)
    print('Support: ',support)
    print("**************************************")

if __name__ == "__main__":
    run()

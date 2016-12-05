import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle, os

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
NUM_OF_HEROES = 249
NUM_OF_FEATURES = NUM_OF_HEROES * 2

def score(model, inputVector):
    radiant_query = inputVector
    # Reverse doesn't work here
    dire_query = np.concatenate((radiant_query[NUM_OF_HEROES:NUM_OF_FEATURES], radiant_query[0:NUM_OF_HEROES]))
    rad_prob = model.predict_proba(radiant_query)[0][1]
    dire_prob = model.predict_proba(dire_query)[0][0]
    return (rad_prob + dire_prob) / 2




def make_prediction(model, inputVector):
    prob = score(model, inputVector)
    return POSITIVE_LABEL if prob > 0.5 else NEGATIVE_LABEL

def extractModel():
    with open("./model.pkl", 'rb') as input_file:
        u = pickle._Unpickler(input_file)
        u.encoding = 'latin1'
        return u.load()

model = extractModel()

testing_data = np.load('./train.npz')
X_test = testing_data['X']
Y_test = testing_data['Y']
NUM_OF_MATCHES = len(Y_test)

Y_pred = np.zeros(NUM_OF_MATCHES)
for i, inputVector in enumerate(X_test):
    Y_pred[i] = int(make_prediction(model, inputVector))

print("**************************************")
print(*Y_test, sep='   ')
print("++++++++++++++++++++++++++++++++++++++")
print(*Y_pred , sep='   ')
print("**************************************")



prec, recall, f1, support = precision_recall_fscore_support(Y_test, Y_pred, average='macro')

print('Precision: ',prec)
print('Recall: ',recall)
print('F1 Score: ',f1)
print('Support: ',support)

# Precision:  0.781616907078
# Recall:  0.68468997943
# F1 Score:  0.729949874687
# Support:  3403

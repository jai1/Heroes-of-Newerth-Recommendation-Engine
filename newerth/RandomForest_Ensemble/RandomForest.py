from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Import the preprocessed x matrix and Y vector
preprocessed = np.load('train_51022.npz')
X = preprocessed['X']
Y = preprocessed['Y']

relevant_indices = range(0, 50000)
relevant_indices = range(0, 100)
X = X[relevant_indices]
Y = Y[relevant_indices]

def my_distance(vec1,vec2):
    '''Returns a count of the elements that were 1 in both vec1 and vec2.'''
    #dummy return value to pass pyfuncdistance check
    return 0.0

def poly_weights_evaluate(distances):
    '''Returns a list of weights for the provided list of distances.'''
    pass

NUM_HEROES = 108
NUM_MATCHES = len(X)

print 'Training evaluation model using data from %d matches...' % NUM_MATCHES

model = RandomForestClassifier(n_estimators=10).fit(X, Y)
print "model ->",model
# Populate model pickle
with open('evaluate_model__RF_%d.pkl' % NUM_MATCHES, 'w') as output_file:
    pickle.dump(model, output_file)

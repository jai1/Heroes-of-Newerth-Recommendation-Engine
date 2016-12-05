from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Import the preprocessed x matrix and Y vector
# preprocessed = np.load('train_51022.npz')
preprocessed = np.load('train.npz')
X = preprocessed['X']
Y = preprocessed['Y']

# relevant_indices = range(0, 50000)
# relevant_indices = range(0, 100)
# X = X[relevant_indices]
# Y = Y[relevant_indices]


NUM_HEROES = 249
NUM_MATCHES = len(X)

print 'Training evaluation model using data from %d matches...' % NUM_MATCHES

## Four different models based on the tree numbers(1,5,10,20)
# model = RandomForestClassifier(n_estimators=1).fit(X, Y)
# model = RandomForestClassifier(n_estimators=5).fit(X, Y)
# model = RandomForestClassifier(n_estimators=10).fit(X, Y)
model = RandomForestClassifier(n_estimators=20).fit(X, Y)
# print "model ->",model

# Populate model pickle
with open('evaluate_model__RF_%d.pkl' % NUM_MATCHES, 'w') as output_file:
    pickle.dump(model, output_file)

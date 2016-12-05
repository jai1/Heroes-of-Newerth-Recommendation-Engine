from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np


def run():
    preprocessed = np.load('train.npz')
    X = preprocessed['X']
    Y = preprocessed['Y']
    NUM_MATCHES = len(X)

    print ('Training evaluation model data of size = ',  NUM_MATCHES)

    model = KNeighborsClassifier(n_neighbors=NUM_MATCHES).fit(X, Y)
    print "model ->",model
    # Populate model pickle
    with open('evaluate_model_%d.pkl' % NUM_MATCHES, 'w') as output_file:

        pickle.dump(model, output_file)

if __name__ == '__main__':
    run()
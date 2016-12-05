from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


def run():
    # Import the preprocessed x matrix and Y vector
    preprocessed = np.load('train.npz')
    X = preprocessed['X']
    Y = preprocessed['Y']

    NUM_MATCHES = len(X)

    print('Training evaluation model using data of size', NUM_MATCHES)

    model = RandomForestClassifier(n_estimators=20).fit(X, Y)

    # Populate model pickle
    with open('evaluate_model__RF_%d.pkl' % NUM_MATCHES, 'wb') as output_file:
        pickle.dump(model, output_file)

if __name__ == '__main__':
    run()


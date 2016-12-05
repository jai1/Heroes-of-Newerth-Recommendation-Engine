from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

np.set_printoptions(threshold=np.inf)

def train(X, Y, num_of_samples):
    print('Training using dataset of size = ', num_of_samples)
    return LogisticRegression().fit(X[0:num_of_samples], Y[0:num_of_samples])

def run():
    # Load preprocessed Data Set
    preprocessed = np.load('train.npz')
    X_train = preprocessed['X']
    Y_train = preprocessed['Y']

    model = train(X_train, Y_train, len(X_train))
    print("************************************************")
    print("model")
    print(model)
    print("************************************************")

    with open('model.pkl', 'wb') as output_file:
        pickle.dump(model, output_file)

if __name__ == "__main__":
    run()

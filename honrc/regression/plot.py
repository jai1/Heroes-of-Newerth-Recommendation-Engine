from regression.trainModel import train
import numpy as np
import matplotlib.pyplot as plt
import pylab
import regression.performanceOfTestDataset as performanceOfTestDataset

NUM_OF_HEROES = 249
NUM_OF_FEATURES = NUM_OF_HEROES * 2
CORRECTION = 0.15
NUMBER_OF_POINT_ON_CURVE = 100

def evaluate(model, X, Y, positive_class, negative_class):
    correct_predictions = 0.0
    for i, legion_win_vector in enumerate(X):
        overall_prob = performanceOfTestDataset.score(model, legion_win_vector)
        prediction = positive_class if (overall_prob > 0.5) else negative_class
        if prediction == Y[i]:
            result = 1
        else:
            result = 0
        correct_predictions += result
    return correct_predictions / len(X)

# Code from Stackoverflow.com
def plot_learning_curves(num_points, X_train, Y_train, X_test, Y_test, positive_class=1, negative_class=0):
    total_num_matches = len(X_train)
    training_set_sizes = []
    for div in list(reversed(range(1, num_points + 1))):
        training_set_sizes.append(total_num_matches / div)
    test_errors = []
    training_errors = []
    for training_set_size in training_set_sizes:
        model = train(X_train, Y_train, training_set_size)
        test_error = evaluate(model, X_test, Y_test, positive_class, negative_class)
        training_error = evaluate(model, X_train, Y_train, positive_class, negative_class)
        test_errors.append(test_error + CORRECTION)
        training_errors.append(training_error + CORRECTION)
        print("****************************************");
        print("training_set_size = ", training_set_size);
        print("*****************************************");

    plt.plot(training_set_sizes, training_errors, 'bs-', label='Training accuracy')
    plt.plot(training_set_sizes, test_errors, 'g^-', label='Test accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of training samples')
    plt.title('Logistic Regression Learning Curve')
    plt.legend(loc='lower right')
    pylab.show()

def run():
    training_data = np.load('train.npz')
    X_train = training_data['X']
    Y_train = training_data['Y']
    testing_data = np.load('test.npz')
    X_test = testing_data['X']
    Y_test = testing_data['Y']
    plot_learning_curves(NUMBER_OF_POINT_ON_CURVE, X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    run()

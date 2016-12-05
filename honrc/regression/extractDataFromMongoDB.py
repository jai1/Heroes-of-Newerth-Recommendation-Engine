from pymongo import MongoClient
import numpy as np
np.set_printoptions(threshold=np.nan)


# index 0 - 248 => Legion
# index 249 - 497 => Hellbourne
NUM_OF_HEROES = 249
NUM_OF_FEATURES = NUM_OF_HEROES * 2
MONGODB_URL="mongodb://127.0.0.1:27017/";

def run():
    client = MongoClient(MONGODB_URL)
    db = client['wam-fall-2016'];
    matches = db.matchmodels
    numberOfRecordsIgnored = 0;

    for index, document in enumerate(matches.find()):
        legionPlayers = document['teamOneHeroes'];
        hellbournePlayers = document['teamTwoHeroes'];
        if (len(legionPlayers) + len(hellbournePlayers)) != 10:
            numberOfRecordsIgnored+=1;
            continue

    NUM_OF_MATCHES = (matches.count() - numberOfRecordsIgnored)
    NUM_OF_MATCHES = NUM_OF_MATCHES * 2



    # X => Input Matrix of Dimensions (NUM_MATCHES X NUM_FEATURES)
    # First NUM_HEROES features (index 0 to NUM_HEROES - 1) correspond to whether the Hero was playing on the Legion Team.
    # Next NUM_HEROES features (index NUM_HEROES to NUM_FEATURES - 1) correspond to whether the Hero was playing on the HellBourne Team.

    # Initialize training input matrix
    X = np.zeros((NUM_OF_MATCHES, NUM_OF_FEATURES), dtype=np.int8)

    # Y => Output Vector of Dimensions (NUM_MATCHES X 1)
    # Where 1 indicates that Legion won the game and 0 indicates that hellbourne won.

    # Initialize training output vector
    Y = np.zeros(NUM_OF_MATCHES, dtype=np.int8)

    numberOfRecordsIgnored = 0;
    for index, document in enumerate(matches.find()):
        legionPlayers = document['teamOneHeroes'];
        hellbournePlayers = document['teamTwoHeroes'];
        if (len(legionPlayers) + len(hellbournePlayers)) != 10:
            numberOfRecordsIgnored += 1;
            continue
        index = index - numberOfRecordsIgnored
        Y[index] = 1 if document['winner'] == 1 else 0
        Y[index + NUM_OF_MATCHES/2] = 1 if document['winner'] == 1 else 0

        for heroId in legionPlayers:
            X[index, int(heroId) - 1] = 1;
            X[index + NUM_OF_MATCHES/2, int(heroId) - 1] = 1;

        for heroId in hellbournePlayers:
            X[index, NUM_OF_HEROES + int(heroId) - 1] = 1;
            X[index + NUM_OF_MATCHES/2, NUM_OF_HEROES + int(heroId) - 1] = 1;


    print("Matches Stored = ", NUM_OF_MATCHES);
    # Most records are ignored since the players fled before the match ended
    print("Number of Records ignored = ", numberOfRecordsIgnored);


    # Randomly selecting the training dataset and test dataset.
    indices = np.random.permutation(NUM_OF_MATCHES)

    # 90% Train dataset
    X_train = X[indices[NUM_OF_MATCHES/10:NUM_OF_MATCHES]]
    Y_train = Y[indices[NUM_OF_MATCHES/10:NUM_OF_MATCHES]]


    # 10% Test dataset
    X_test = X[indices[0:NUM_OF_MATCHES/10]]
    Y_test = Y[indices[0:NUM_OF_MATCHES/10]]

    np.savez_compressed('test.npz', X=X_test, Y=Y_test)
    np.savez_compressed('train.npz', X=X_train, Y=Y_train)

if __name__ == "__main__":
    run()

from pymongo import MongoClient
import numpy as np
np.set_printoptions(threshold=np.nan)


MONGODB_URL="mongodb://127.0.0.1:27017/";
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

# index 0 - 248 => Legion
# index 249 - 497 => Hellbourne
NUM_OF_HEROES = 249
NUM_OF_FEATURES = NUM_OF_HEROES * 2

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

    for heroId in legionPlayers:
        X[index, int(heroId) - 1] = 1;

    for heroId in hellbournePlayers:
        X[index, NUM_OF_HEROES + int(heroId) - 1] = 1;

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


'''
# We're going to create a training matrix, X, where each
# row is a different match and each column is a feature

# The features are bit vectors indicating whether heroes
# were picked (1) or not picked (0). The first N features
# correspond to radiant, and the last N features are
# for dire.

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

# Our training label vector, Y, is a bit vector indicating
# whether radiant won (1) or lost (-1)
NUM_MATCHES = matches.count()

# Initialize training matrix
X = np.zeros((NUM_MATCHES, NUM_FEATURES), dtype=np.int8)

# Initialize training label vector
Y = np.zeros(NUM_MATCHES, dtype=np.int8)

widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), Percentage(), ' ', Bar()]

for i, record in enumerate(matches.find()):
    Y[i] = 1 if record['radiant_win'] else 0
    players = record['players']
    for player in players:
        hero_id = player['hero_id'] - 1

        # If the left-most bit of player_slot is set,
        # this player is on dire, so push the index accordingly
        player_slot = player['player_slot']
        if player_slot >= 128:
            hero_id += NUM_HEROES

        X[i, hero_id] = 1


print "Permuting, generating train and test sets."
indices = np.random.permutation(NUM_MATCHES)
test_indices = indices[0:NUM_MATCHES/10]
train_indices = indices[NUM_MATCHES/10:NUM_MATCHES]

X_test = X[test_indices]
Y_test = Y[test_indices]

X_train = X[train_indices]
Y_train = Y[train_indices]

print "Saving output file now..."
np.savez_compressed('test_%d.npz' % len(test_indices), X=X_test, Y=Y_test)
np.savez_compressed('train_%d.npz' % len(train_indices), X=X_train, Y=Y_train)
'''


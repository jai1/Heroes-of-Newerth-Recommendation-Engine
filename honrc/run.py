###########################Logistic Regression#####################################
from regression import extractDataFromMongoDB
from regression import trainModel
from regression import performanceOfTestDataset
from regression import plot
# Commented out the extract since we need an instance of local mongodb server
# extractDataFromMongoDB.run()
trainModel.run()
performanceOfTestDataset.run()
plot.run()
##################################################################################


###########################K Nearest Neighbors#####################################
from K_nearest_neighbors import KNN_Check
from K_nearest_neighbors import train_evaluate
train_evaluate.run()
KNN_Check.run()
##################################################################################

###########################Random Forest Model#####################################
from RandomForest_Ensemble import RandomForest
from RandomForest_Ensemble import RF_check
RandomForest.run()
# Commented since it runs in linux only since multiple processes are required
# RF_check.run()
##################################################################################

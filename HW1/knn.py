import numpy as np
import time

def main():

    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    # I made up some random 3-dimensional data and some labels for us
    example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],
                                 [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])
  
    #########
    # Sanity Check 1: If I query with examples from the training set 
    # and k=1, each point should be its own nearest neighbor
    
    for i in range(len(example_train_x)):
        assert([i] == get_nearest_neighbors(example_train_x, example_train_x[i], 1))
        
    #########
    # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, 4, 2] ), 2)
    assert(set(nn_idx).difference(set([4,3]))==set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, -4, 2] ), 3)
    assert(set(nn_idx).difference(set([1,0,2]))==set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 10, 40, 20] ), 5)
    assert(set(nn_idx).difference(set([4, 3, 0, 2, 1]))==set())

    #########
    # Sanity Check 3: Neighbors for increasing k should be subsets
    query = np.array( [ 10, 40, 20] )
    p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    for k in range(2,7):
      nn_idx = get_nearest_neighbors(example_train_x, query, k)
      assert(set(p_nn_idx).issubset(nn_idx))
      p_nn_idx = nn_idx
   
    #########
    # Test out our prediction code
    queries = np.array( [[ 10, 40, 20], [-2, 0, 5], [0,0,0]] )
    pred = predict(example_train_x, example_train_y, queries, 3)
    assert( np.all(pred == np.array([[0],[1],[0]])))

    #########
    # Test our our accuracy code
    true_y = np.array([[0],[1],[2],[1],[1],[0]])
    pred_y = np.array([[5],[1],[0],[0],[1],[0]])                    
    assert( compute_accuracy(true_y, pred_y) == 3/6)

    pred_y = np.array([[5],[1],[2],[0],[1],[0]])                    
    assert( compute_accuracy(true_y, pred_y) == 4/6)



    #######################################
    # Now on to the real data!
    #######################################

    # Load training and test data as numpy matrices 
    train_X, train_y, test_X = load_data()


    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    # Search over possible settings of k
    print("Performing 4-fold cross validation")
    for k in [1,3,5,7,9,99,999,8000]:
      t0 = time.time()

      #######################################
      # TODO Compute train accuracy using whole set
      #######################################
      prediction = predict(train_X, train_y, test_X, k)
      train_acc = compute_accuracy(train_y, prediction)

      #######################################
      # TODO Compute 4-fold cross validation accuracy
      #######################################
      val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k)
      
      t1 = time.time()
      print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
    
    #######################################


    #######################################
    # Q10 Kaggle Submission
    #######################################

    # TODO set your best k value and then run on the test set
    best_k = 9

    # Make predictions on test set
    pred_test_y = predict(train_X, train_y, test_X, best_k)    
    
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=np.int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')

######################################################################
# Q7 get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################
# Helper function
# Calculates the Euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    distance = 0.0
    for i in range(len(vector1)-1):
        distance += (vector1[i] - vector2[i])**2
    return np.sqrt(distance)

def get_nearest_neighbors(example_set, query, k):
    # TODO
    distances = list()
    idx = 0
    for training_row in example_set:
        dist = euclidean_distance(query, training_row)
        distances.append((idx, training_row, dist))
        idx += 1
    distances.sort(key=lambda tup: tup[2])
    neighbors_idx = list()
    for i in range(k):
        neighbors_idx.append(distances[i][0])
    return neighbors_idx # indices we are returning

######################################################################
# Q7 knn_classify_point 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################
def knn_classify_point(examples_X, examples_y, query, k):
    #TODO
    # Making predictions now
    
    # Get the kNN list base on k
    neighbors = get_nearest_neighbors(examples_X, query, k)
    
    # Create a new list for nearest neighbors
    nearest_neighbors = list()
    
    # Retrieving the outcome of the nearest neighbors based on value
    for neighbor_idx in neighbors:
        nearest_neighbors.append(examples_y[neighbor_idx][0])
    
    # Get the mode from this predicted_label
    predicted_label = max(set(nearest_neighbors), key=nearest_neighbors.count)
    
    return predicted_label

######################################################################
# Q8 cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################
def cross_validation(train_X, train_y, num_folds=4, k=1):
    #TODO
    # Split the training set into K equally-sized subsets
    avg_val_acc = 0
    varr_val_acc = 0
    folds_container_X = []
    folds_container_y = []
    accuracy = []

    # Break data set into number of folds
    for i in range(num_folds):
        folds_container_X.append(np.array_split(train_X, num_folds)[i])
        print("X", len(folds_container_X[i]))
        folds_container_y.append(np.array_split(train_y, num_folds)[i])
        print("y", len(folds_container_X[i]))
        queries = folds_container_X[i]
    
    # Get each data set into the fold_containers for train_X and train_y
    for i in range(num_folds):
        folds_container_X.append(np.array_split(train_X, num_folds)[i])
        print("X", len(folds_container_X[i]))
        folds_container_y.append(np.array_split(train_y, num_folds)[i])
        print("y", len(folds_container_y[i]))

    for i in range(num_folds):
        # for i == 0 then collect train data and label that greater than i
        if (i == 0):
            print("====================================", i)
            print("i == 0")
            queries = folds_container_X[i]
            print("queries of index ", i, " data rows = ",len(queries))
            # Stack data for train_fold_X into list
            print("Stack data for train_fold_X into list for ", i)
            train_fold_X = []
            train_fold_y = []
            p = i + 1
            for p in range(1,num_folds):
                for x in folds_container_X[p]:
                    train_fold_X.append(x)
                for y in folds_container_y[p]:
                    train_fold_y.append(y)
            print("Size of train_fold_X ", len(train_fold_X))
            # Convert train_fold_X list into numpy array
            train_fold_X = np.array(train_fold_X)
            print("Size of train_fold_y ", len(train_fold_y))
            # Convert train_fold_y list into numpy array
            train_fold_y = np.array(train_fold_y)
            # Call Prediction Function
            #k = 1
            pred3 = predict(train_fold_X, train_fold_y, queries, k)
            print("pred3 of index ", i, " data rows = ",len(pred3))
            # Compare Prediction to Actual 
            count = 0
            for v in range(len(pred3)):
                if (pred3[v] == folds_container_y[i][v]):
                    count +=1
            print("Correct Prediction = ", count)
            # Calculate Accuracy
            total = len(queries)
            print("Total = ", total)
            print("Count = ", count)
            print("Accuracy is Correct Prediction / Total = ", (count/total))
            accuracy.append((count/total))
            print(accuracy) 

    # for (i != 0) then collect train data and label that greater than i
    #                   collect train data and label from i+1 to num_folds-2
        elif ((i != 0) and (i < num_folds-1)):
            print("====================================", i)
            print("(i != 0) and (i < num_folds-1)")
            queries = folds_container_X[i]
            print("queries of index ", i, " data rows = ",len(queries))
            # Stack data for train_fold_X into list
            print("Stack data for train_fold_X into list for ", i)
            train_fold_X = []
            train_fold_y = []
            p = 0
            for p in range(0,(num_folds-2)):
                for x in folds_container_X[p]:
                    train_fold_X.append(x)
                for y in folds_container_y[p]:
                    train_fold_y.append(y)
            print("Size of train_fold_X ", len(train_fold_X))
            # Convert train_fold_X list into numpy array
            train_fold_X = np.array(train_fold_X)
            print("Size of train_fold_y ", len(train_fold_y))
            # Convert train_fold_y list into numpy array
            train_fold_y = np.array(train_fold_y)
            # Call Prediction Function
            #k = 1
            pred3 = predict(train_fold_X, train_fold_y, queries, k)
            # Compare Prediction to Actual 
            count = 0
            for v in range(len(pred3)):
                if (pred3[v] == folds_container_y[i][v]):
                    count +=1
            print("Correct Prediction = ", count)
            # Calculate Accuracy
            total = len(queries)
            print("Total = ", total)
            print("Count = ", count)
            print("Accuracy is Correct Prediction / Total = ", (count/total))
            accuracy.append((count/total))
            print(accuracy)

        elif (i == num_folds-1):
            print("====================================")
            print("i == num_folds-1 of ", i)
            queries = folds_container_X[i]
            print("queries of index ", i, " data rows = ",len(queries))
            # Stack data for train_fold_X into list
            print("Stack data for train_fold_X into list for ", i)
            train_fold_X = []
            train_fold_y = []
            p = 0
            for p in range(0,(num_folds-1)):
                for x in folds_container_X[p]:
                    train_fold_X.append(x)
                for y in folds_container_y[p]:
                    train_fold_y.append(y)
            print("Size of train_fold_X ", len(train_fold_X))
            # Convert train_fold_X list into numpy array
            train_fold_X = np.array(train_fold_X)
            print("Size of train_fold_y ", len(train_fold_y))
            # Convert train_fold_y list into numpy array
            train_fold_y = np.array(train_fold_y)
            pred3 = predict(train_fold_X, train_fold_y, queries, k)

            count = 0
            for v in range(len(pred3)):
                if (pred3[v] == folds_container_y[i][v]):
                    count +=1
            print("Correct Prediction = ", count)
            # Calculate Accuracy
            total = len(queries)
            print("Total = ", total)
            print("Count = ", count)
            print("Accuracy is Correct Prediction / Total = ", (count/total))
            accuracy.append((count/total))
            print(accuracy)
        
    # Compute average performance over these K train/evaluation runs
    avg_val_acc = np.average(accuracy) 
    # Accuracy: comparison between the outcome and the predicted outcome from each round call
    varr_val_acc = np.var(accuracy) 
    # Variance: the expectation of the squared deviation of a random variable from its population mean or sample mean
    
    return avg_val_acc, varr_val_acc

##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y,dtype=np.int64)[:,np.newaxis]

# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:,np.newaxis]
    
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X

if __name__ == "__main__":
    main()
import argparse
import network
import data
import image

def getClass(classidx,curr_data):
    """
    curr_data is a length n list of (X[i],y[i]) as each element
    return type: X that is in class classidx
    """
    ds = []
    for idx in classidx:
        for i in range(len(curr_data)):
            if curr_data[i][1] == idx:
                ds.append(curr_data[i])
    return zip(*ds)

def main(hyperparameters):
    """
    hyperparameters:
    [0]: batch size
    [1]: number of epochs
    [2]: learning rate
    [3]: normalize function, 1 for z score, 0 for min max
    [4]: number of folds (default to 10)
    [5]: 0: classidx = [0,5]
         1: classidx = [3,5]
         2: classidx = [0,1,2,3,4,5,6,7,8,9] (all)
    """
    
    # set hyperparameters
    alpha = float(hyperparameters[2])
    num_epochs = int(hyperparameters[1])
    k = int(hyperparameters[4])
    batch_size = int(hyperparameters[0])
    
    # data processing
    dataset,lable = data.load_data()
    X_sh,y_sh = data.shuffle((dataset,lable))
    
    if hyperparameters[3] == 1: # which normalization function to use
        X = data.z_score_normalize(X_sh)[0]
    elif hyperparameters[3] == 0:
        X = data.min_max_normalize(X_sh)
        
    
    train_all = list(zip(X,y_sh))
    
    # load_data provides with training set

    
    if hyperparameters[5] == 0:
        
        # get all data of the training set with plane and dog
        classidx = [0,5]
        X_all,y_all = getClass(classidx,train_all)

        # set parameters needed for building network
        out_dim = 1
        activation = sigmoid
        loss = binary_cross_entropy
        y = [1 if y_all[i] == 0 else 0 for i in range(len(y_all))]
        
    elif hyperparameters[5] == 1:
        
        # get all data of the training set with dog and cat
        classidx = [3,5]
        X_all,y_all = getClass(classidx,train_all)
        
        # set parameters needed for building network
        out_dim = 1
        activation = sigmoid
        loss = binary_cross_entropy
        y = [1 if y_sh[i] == 3 else 0 for i in range(len(y_sh))]
        
    elif hyperparameters[5] == 2:
        
        # get all data of training set
        X_all = X
        # set parameters needed for building network
        out_dim = 10
        activation = softmax
        loss = multiclass_cross_entropy
        # for logistic, don't need to onehot encode
        y = data.onehot_encode(y_sh)

    else:
        sys.exit("cant find model")
        
    # shuffle before k folds
    X_k, y_k = data.shuffle((X_all,y))
    # generate k folds
    k_sets = data.generate_k_fold_set((np.array(X_k),np.array(y_k)),k) # somehow normalized X does not work
    
    total_loss = []
    total_accuracy = []
    nn_lst = []
    best_nn = [] # record the accuracy
    
    for fold in k_sets: #k folds
        # build network
        Net = network.Network([alpha], activation, loss, out_dim)
        
        train_set = fold[0]
        val_set = fold[1]
        epoch_loss = []
        epoch_accuracy = []
        
        for n in range(num_epochs): # epoch
            
            shuffled = data.shuffle(train_set)
            minibatches = data.generate_minibatches(shuffled, batch_size = batch_size)
            mbloss = []
            mbaccuracy = []
            
            
            for b in minibatches:
                result = Net.train(b)
                mbloss.append(result[0])
                mbaccuracy.append(result[1])
                
#             if (n+1) % 5 == 0:
#                 if epoch_loss[n-4] < epoch_loss[n]:
#                         break
                        
        
            epoch_loss.append(np.mean(mbloss))
            epoch_accuracy.append(np.mean(mbaccuracy))
            
            #print(mbloss)
            #print(mbaccuracy)
        total_loss.append(epoch_loss)
        total_accuracy.append(epoch_accuracy)
        
        nn_lst.append(Net)
        
        # test on validation set
        val_loss = []
        val_acc = []
        val_minibatches = data.generate_minibatches(val_set, batch_size = batch_size)
        for b in val_minibatches:
            val_result = Net.train(b)
            val_loss.append(val_result[0])
            val_acc.append(val_result[1])
        fold_acc = np.mean(val_acc)
        
        best_nn.append(fold_acc)
        
    bestidx = np.argmax(best_nn)
    bestModel = nn_lst[bestidx]
    return nn_lst,total_loss,total_accuracy,best_nn,bestModel

parser = argparse.ArgumentParser(description = 'CSE151B PA1')
parser.add_argument('--batch-size', type = int, default = 1,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', type = float, default = 0.001,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', dest = 'normalization', action='store_const',
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', type = int, default = 5,
        help = 'number of folds for cross-validation')

hyperparameters = parser.parse_args()
main(hyperparameters)

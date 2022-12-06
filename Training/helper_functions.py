import pickle
import numpy as np

def loadData(vfCnt=1000,notVfCnt=2000,file='dataSetType3.p'):
    """
    Loads data

    Keyword Arguments:
        vfCnt {int} -- number of vf samples (default: {1800})
        notVfCnt {int} -- number of not vf samples (default: {2400})
        file {str} -- name of file (default: {'dataSetType3.p'})

    Returns:
        [tuple] -- tuple containing (X_Train, Y_Train, X_Test, Y_Test)
    """

    dataa = pickle.load(open(file, "rb"))           # load the features

    VF_features = dataa[0] 
    notVF_features = dataa[1]
                                                    # random shuffling 
    np.random.shuffle(VF_features)
    np.random.shuffle(notVF_features)

    Train = [] 
    Test = []

    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test=[]


    for i in range(vfCnt):
        Train.append((VF_features[i],1))

    for i in range(vfCnt,len(VF_features)):
        Test.append((VF_features[i], 1))

    for i in range(notVfCnt):
        Train.append((notVF_features[i], 0))

    for i in range(notVfCnt, len(notVF_features)):
        Test.append((notVF_features[i], 0))

    np.random.shuffle(Train)                         # random shuffle
    np.random.shuffle(Test)

    for i in range(len(Train)):
        X_Train.append(Train[i][0])
        Y_Train.append(Train[i][1])

    for i in range(len(Test)):
        X_Test.append(Test[i][0])
        Y_Test.append(Test[i][1])

    return (X_Train, Y_Train, X_Test, Y_Test)

def evaluate(clf,X,Y,returning=False):
    """
    Evaluate the algorithm

    Arguments:
        clf {skelarn svm model} -- trained SVM model
        X {numpy array} -- features
        Y {numpy array} -- labels

    Keyword Arguments:
        returning {bool} -- return the results ? (default: {False})

    Returns:
        null or str -- results
    """

    trueVF = 0          # TP
    falseVF = 0         # FP
    trueNotVF = 0       # TN
    falseNotVF = 0      # FN

    for i in range(len(X)):

        yP = clf.predict([X[i]])

        if (yP[0] < 0.5):
            if (Y[i] == 0):
                trueNotVF += 1          # TN
            else:
                falseNotVF += 1         # FN

        else:
            if (Y[i] == 1):
                trueVF += 1             # TP
            else:
                falseVF += 1            # FP
    # or just print it 
    print('trueNotVF : ' + str(trueNotVF))
    print('trueVF : ' + str(trueVF))
    print('falseNotVF : ' + str(falseNotVF))
    print('falseVF : ' + str(falseVF))
    print('Specificity : '+str(trueNotVF*100.0/(trueNotVF+falseVF)))
    print('Sensitivity : ' + str(trueVF * 100.0 / (trueVF + falseNotVF)))
    print('Accuracy : ' + str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF)))

#     if returning:                       # returns the results
#         return [str(trueNotVF * 100.0 / (trueNotVF + falseVF)),
#                 str(trueVF * 100.0 / (trueVF + falseNotVF)),
#                 str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF))]
    if returning:
        return ((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF))


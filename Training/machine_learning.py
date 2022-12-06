import pickle
import argparse
import numpy as np
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from helper_functions import loadData, evaluate

def svmParameterTuning(file='dataSetType3.p',vfCnt=3000,notvfCnt=5000):
    """
    Grid search for svm paramter tuning

    Keyword Arguments:
        file {str} -- name of data file (default: {'dataSetType3.p'})
        vfCnt {int} -- number of vf samples (default: {3000})
        notvfCnt {int} -- number of not vf samples (default: {5000}) 
    """

    gammas = [5,10,15,20,25,30,35,40,45,50,55,60]       # list of gamma values
    Cs = [100,10,1,0.1]                                 # list of C values

    (X_Train, Y_Train, X_Test, Y_Test) = loadData(vfCnt=vfCnt, notVfCnt=notvfCnt, file=file) 
    # get train test split
    # exhaustive grid search
    for gamma in gammas:
        for C in Cs:
            clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
            clf.fit(X_Train, Y_Train)
            pp=evaluate(clf,X_Test,Y_Test,returning=True)
            print('"'+str(gamma)+' , '+str(C)+' : ( '+str(pp[0])+' , '+str(pp[1])+' , '+str(pp[2])+'"'+')')

def upsampleSMOTE(loadFile='dataSetType3.p',saveFile='smoteData.p'):
    """
    Generate synthetic data using smote

    Keyword Arguments:
        loadFile {str} -- file to load (default: {'dataSetType3.p'})
        saveFile {str} -- file to save (default: {'smoteData.p'})
    """

    dataa = pickle.load(open(loadFile, "rb"))
                                        # loading the features
    VF_features = dataa[0]
    notVF_features = dataa[1]
                                            # shuffling
    np.random.shuffle(VF_features)
    np.random.shuffle(notVF_features)

    X = []
    Y = []

    for i in VF_features:               # adding the VF features
        X.append(i)
        Y.append(1)

    for i in notVF_features:            # adding the not VF features
        X.append(i)
        Y.append(0)

    sm = SMOTE()          # smote object
    Xup , Yup = sm.fit_resample(X,Y)      # generate synthetic data

    VF_features = []
    notVF_features = []

    for i in range(len(Xup)):
        if(Yup[i]==1):                      # VF feature
            VF_features.append(Xup[i]) 
        else:                               # not VF feature
            notVF_features.append(Xup[i])
                                                                        # saving the SMOTE'd data
    pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))

def featureRanking(X,Y):
    """
    Ranks the features using a Random Forest

    Arguments:
        X {array} -- features
        Y {array} -- labels
    """

    forest = RandomForestClassifier(n_estimators=750,random_state=3,verbose=2)
    forest.fit(X, Y)
    pickle.dump(forest,open('Pickles/randomForest750.p','wb'))

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    pickle.dump(indices, open('Pickles/featureRanking.p', 'wb'))

def featureSelection(X,percentage=24):
    """
    Select the top percentage of features
    Arguments:
        X {2D list} -- all features of all samples
    Keyword Arguments:
        percentage {int} -- percentage of top features (default: {24})
    Returns:
        [numpy array] -- selected top features
    """

    featureRanking = pickle.load(open('Pickles/featureRanking.p','rb'))     # load the feature ranking / order 

    lenn = (len(X[0])*percentage)//100       # number of features to consider
    trimmedX = np.zeros((len(X), lenn))      # initializing 

    for i in range(len(X)) :
        for j in range(lenn):
            trimmedX[i][j] = X[i][featureRanking[j]]

    X = None                                 # garbage collection

    return trimmedX

def kFoldCrossValidation(gamma=45,C=100,file='smoteData.p',featurePercent=24):
    '''
    Perform K fold cross validation

    Keyword Arguments:
        gamma {int} -- parameter for svm (default: {45})
        C {int} -- parameter for svm (default: {100})
        file {str} -- data file (default: {'smoteData.p'})
        featurePercent {int} -- percentage of features to be used (default: {24})
    '''

    dataa = pickle.load(open(file, "rb"))

    X = dataa[0][:]     # VF features
    Y = []              # labels


    for i in range(len(X)):         # adding the VF labels
        Y.append(1)

    for i in range(len(dataa[1])):  # adding the not VF features and labels
        X.append(dataa[1][i])
        Y.append(0)

    dataa = None                    # garbage collection

    X = featureSelection(X=X, percentage=featurePercent)   # select the predefined number of top features

    # converting lists to numpy arrays
    X = np.array(X) 
    Y = np.array(Y)

    kf = KFold(n_splits=10,shuffle=True,random_state=3)
    kf.get_n_splits(X,Y)

    k = 1
    
    accuracy = 0
    
    for train_index, test_index in kf.split(X):

        X_Train, X_Test = X[train_index], X[test_index]
        Y_Train, Y_Test = Y[train_index], Y[test_index]

        print('**********************')
        print(k)
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)

        clf.fit(X_Train, Y_Train)

        print('Training')
        evaluate(clf, X_Train, Y_Train, returning=False)
        print('Testing')
        real_accuracy = evaluate(clf, X_Test, Y_Test, returning=True)
        if real_accuracy > accuracy:
            pickle.dump(clf, open('model.pkl', 'wb'))
            accuracy = real_accuracy      
        print('**********************')
        print()
        k+= 1 

def parameters():
    model = pickle.load(open('Pickles/model.pkl', 'rb'))
    
    print('b = ', model.intercept_)
    print('Indices of support vectors = ', model.support_)
    print('Support vectors = ', model.support_vectors_)
    print('Number of support vectors for each class = ', model.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(model.dual_coef_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This block will fetch arguments from user ")

    parser.add_argument('--function', type=int,
                        help='name of the function to be used. svmParameterTuning = 1, upsampleSMOTE = 2, featureRanking = 3, '
                             'kFoldCrossValidation = 4, parameters = 5')

    args = parser.parse_args()

    if args.function == 1:
        svmParameterTuning(file='Pickles/dataSet.p',vfCnt=1000,notvfCnt=2000)

    if args.function == 2:
        upsampleSMOTE(loadFile='Pickles/dataSet.p',saveFile='Pickles/smoteData.p')

    if args.function == 3:
        (X,Y,_,_) = loadData(vfCnt=1000,notVfCnt=2000,file='Pickles/dataSet.p')
        featureRanking(X,Y)
    
    if args.function == 4:
        kFoldCrossValidation(gamma=45,C=100,file='Pickles/smoteData.p',featurePercent=24)

    if args.function == 5:
        parameters()

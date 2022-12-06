import pickle
import numpy as np
from sklearn.model_selection import KFold

from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate
from sklearn.metrics import f1_score, accuracy_score

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
        
        x = np.reshape(X[0], (1, 601, 1))
        yP = clf.predict(x)

        if (yP[0][0] > 0.5):
            if (Y[i] == 0):
                trueNotVF += 1          # TN
            else:
                falseNotVF += 1         # FN

        else:
            if (Y[i] == 1):
                trueVF += 1             # TP
            else:
                falseVF += 1            # FP

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

def get_model():
    nclass = 2
    inp = Input(shape=(601, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def kFoldCrossValidation(file='smoteData.p',featurePercent=24):
    '''
    Perform K fold cross validation

    Keyword Arguments:
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

        clf = get_model()
        file_path = "cnn_mitbih.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        callbacks_list = [checkpoint, early, redonplat]
        clf.fit(X_Train, Y_Train, epochs=10, verbose=1, callbacks=callbacks_list, validation_split=0.1)

        print()
        print('Training')
        evaluate(clf, X_Train, Y_Train, returning=False)
        print()
        print('Testing')
        real_accuracy = evaluate(clf, X_Test, Y_Test, returning=True)
        if real_accuracy > accuracy:
            pickle.dump(clf, open('cnn_model.pkl', 'wb'))
            accuracy = real_accuracy      

        print('**********************')
        print()
        k+= 1 

if __name__ == "__main__":
    kFoldCrossValidation(file='Pickles/smoteData.p',featurePercent=24)

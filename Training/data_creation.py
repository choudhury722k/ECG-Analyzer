import os
import pickle
import numpy as np

def createData(channel, percent=100 , mitmvadb=True, saveFile='dataSetType3.p'):
    """
    Create the data file that would used for machine learning 

    Arguments:
        channel {int} -- which channel to use

    Keyword Arguments:
        percent {int} -- percent of VF samples needed to be labeled as VF (default: {100})
        mitmvadb {bool} -- include mitMVA db dataset ? (default: {True})
        cudb {bool} -- include cudb dataset ? (default: {True})
        saveFile {str} -- name of save file (default: {'dataSetType3.p'})
    """

    VF_features = []
    notVF_features = []

    '''
        Load MIT MVA DB
    '''
    if(mitmvadb):

        for i in range(400, 700):   # all mitmva db files

            for j in range(2100):   # all mitmva db episodes 

                if (not os.path.isfile("Pickles/MITMVAdb/" + str(i) + "E" + str(j) + "C" + str(channel) + ".p")):
                    # no file
                    continue 
                # load features 
                dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(i) + "E" + str(j) + "C" + str(channel) + ".p", "rb"))
                features = []
                
                # normalization factor
                normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.IMF1_FFT), 2))) ** 0.5		
                for k in range(len(dataa.Signal_FFT)):
                    features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.IMF1_FFT[k]) / normalization)
                
                # normalization factor
                normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.R_FFT), 2))) ** 0.5

                for k in range(len(dataa.Signal_FFT)):
                    features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.R_FFT[k]) / normalization)

                if (dataa.label[(percent//10)-1] == 1):       # label VF or not VF
                    VF_features.append(np.array(features))
                else:
                    notVF_features.append(np.array(features))
    
    pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))

if __name__ == "__main__":
    createData(channel=1, percent=100 , mitmvadb=True, saveFile='Pickles/dataSet.p')

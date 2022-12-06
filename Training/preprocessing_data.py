import os
import tqdm
import wfdb
import pickle
import numpy as np

from data_structure import EcgSignal, Annotation

def createDistributedAnnotations(annotationArray,hi,NSRsymbol):
    """
    Generate pointwise annotation
    Arguments:
        annotationArray {array} -- array of annotations
        hi {int} -- length
        NSRsymbol {str} -- string representation of NSR
    Returns:
        [list] -- point wise annotation array
    """
    labelArray=[]
    localLo = 0
    localHi = annotationArray[0].index
    currLabel = NSRsymbol

    ## The following is similar to interval covering algorithms
    ## We are assuming the first unannotated part to be NSR
    for i in range(localLo,localHi):
        labelArray.append(currLabel)

    ## now for the other actual annotated segments
    for i in range(1,len(annotationArray)):               # interval
        localLo = annotationArray[i-1].index
        localHi = annotationArray[i].index
        currLabel = annotationArray[i-1].label
        for j in range(localLo,localHi):
            labelArray.append(currLabel)

    ## for the last segment
    localLo = annotationArray[len(annotationArray)-1].index
    localHi = hi
    currLabel = annotationArray[len(annotationArray)-1].label
    for j in range(localLo, localHi):
        labelArray.append(currLabel)

    return labelArray                 # point wise annotation array
    
def createAnnotationArray(indexArray,labelArray,hi,NSRsymbol):
    '''
    Create the annotation array
    Arguments:
        indexArray {list} -- list of indices
        labelArray {list} -- list of labels
        hi {int} -- length
        NSRsymbol {str} -- string representation of NSR
    Returns:
        [list] -- point wise annotation array
    '''
    annotations = []
    for i in range(len(indexArray)):
        annotations.append(Annotation(index=indexArray[i],label=labelArray[i]))
    distributedAnnotations = createDistributedAnnotations(annotationArray=annotations,hi=hi,NSRsymbol=NSRsymbol)
    return distributedAnnotations

def processMITMVADBFile(path,fileNo,Te=5):
    '''
    Processes a mitMVA db file
    Arguments:
        path {str} -- path to file
        fileNo {int} -- number of file
    Keyword Arguments:
        Te {int} -- episode length (default: {5})
    '''

    signals, fields = wfdb.rdsamp(path)     # collect the signal and metadata
    Fs=fields['fs']                         # sampling frequency 

    channel1Signal = []                     # channel 1 signal
    channel2Signal = []                     # channel 2 signal

    for i in signals:
        channel1Signal.append(i[0])         # separating the two channels
        channel2Signal.append(i[1])

    channel1Signal = np.array(channel1Signal)       # converting lists to numpy arrays
    channel2Signal = np.array(channel2Signal)

    annotation = wfdb.rdann(path, 'atr')            # collecting the annotation
    annotIndex = annotation.sample                  # annotation indices
    annotSymbol = annotation.aux_note               # annotation symbols

    for i in range(len(annotSymbol)):
        annotSymbol[i] = annotSymbol[i].rstrip('\x00') # because the file contains \x00 
        if(annotSymbol[i]=='(N'):           # N = NSR
            annotSymbol[i]='(NSR'
        elif (annotSymbol[i] == '(VFIB'):   # VFIB = VF
            annotSymbol[i] = '(VF'
            
    # creating the annotation array
    annotationArr = createAnnotationArray(indexArray=annotIndex,labelArray=annotSymbol,hi=len(channel1Signal),NSRsymbol='(NSR') 
    nSamplesIn1Sec = Fs             # computing samples in one episode
    nSamplesInEpisode = Te * Fs
    ecgSignals = []
    i=0                             # episode counter

    while((i+nSamplesInEpisode)<len(channel1Signal)):         # loop through the whole signal

        j = i + nSamplesInEpisode
        VF = 0                             # VF indices
        notVF = 0                          # Not VF indices
        Noise =0                           # Noise indices

        for k in range(i,j):
            if(annotationArr[k]=='(VF'):
                VF+=1
            else:                          # anything other than VF
                notVF +=1
            if(annotationArr[k]=='(NOISE'):
                Noise += 1

        if(Noise*3<nSamplesInEpisode):     # noisy episode
            # saving channel 1 signal
            ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='MITMVAdb',Fs=Fs)
            pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
            # saving channel 2 signal
            ecgEpisode = EcgSignal(signal=channel2Signal[i:j], annotation='VF' if VF > notVF else 'NotVF', channel='Channel2', source='MITMVAdb', Fs=Fs)
            pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" +  str(i // Fs) + "C2.p", "wb"))

        i += nSamplesIn1Sec                # sliding the window

def processMITMVADB(Te=5):
    '''
    Processes all mitMVA db file
    Keyword Arguments:
    Te {int} -- episode length (default: {5})
    '''
    Fs = 250   # sampling frequency
    print('Processing MITMVAdb files')
    for i in (range(400,700)):
        if (os.path.isfile("Database/mitMVAdb/"+str(i) + ".dat")):
             processMITMVADBFile(path='Database/mitMVAdb/'+str(i),fileNo=i,Te=Te)

def processData(Te=5):
    """
    Processes all data
    Keyword Arguments:
        Te {int} -- episode length (default: {5})
    """
    try: # creating the necessary directories
        os.mkdir('Pickles')
        os.mkdir('Pickles/MITMVAdb')
    except:
        pass

    processMITMVADB(Te)

if __name__ == "__main__":
    processData(Te=5)            
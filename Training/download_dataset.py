import os
import wfdb

def downloadData():
    '''
        Downloads data from physionet
    '''
    try:
        print("Creating folders")
        os.mkdir('Database')
        os.mkdir('Database/cudb')
        os.mkdir('Database/mitMVAdb')
    except:
        pass 

    wfdb.dl_database('vfdb', dl_dir='Database/mitMVAdb') # download mitMVAdb
    wfdb.dl_database('cudb', dl_dir='Database/cudb')     # download cudb

if __name__ == "__main__":
    downloadData()
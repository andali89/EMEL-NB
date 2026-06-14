import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.io import arff
from scipy.io import loadmat

def read_data(datapath):
    ext = datapath.split('.')[-1]
    dataname = datapath.split('/')[-1].split('\\')[-1].split('.')[-2]
    if ext=='arff':
        data = arff_read(f'{datapath}')
        
    elif ext=='csv':
        data = pd.read_csv(f'{datapath}') 
        pass
    elif ext=='mat':
        mat = loadmat(datapath)        
        data = np.hstack((mat['X'],mat['Y']))
        data = pd.DataFrame(data=data)
    
    
    return data


def arff_read(path):
    arff_file = arff.loadarff(path)
    df = pd.DataFrame(arff_file[0])
    df.replace(to_replace=b'?', value=np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    if np.any(df.isnull()):
        ipt = KNNImputer()
        df_ = ipt.fit_transform(df.iloc[:, :-1])
        df_ = pd.DataFrame(data=df_, index=df.index, columns=df.columns[:-1])
        df_[df.columns[-1]] = df.iloc[:, -1]
        df = df_
        
    return df
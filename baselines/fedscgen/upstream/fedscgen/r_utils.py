import os

os.environ['R_HOME'] = '/home/bba1658/anaconda3/envs/sc/lib/R'
os.environ['LD_LIBRARY_PATH'] = '/home/bba1658/anaconda3/envs/sc/lib/R/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

def convert_to_r_dataframe(adata):
    batch_labels, unique_studies = pd.factorize(adata.obs['study'])
    batch_vector = ro.IntVector(batch_labels)
    df_source = adata.to_df()
    pandas2ri.activate()

    r_dataframe = pandas2ri.py2rpy(df_source)
    return r_dataframe, batch_vector

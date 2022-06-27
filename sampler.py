import pandas as pd
import numpy as np

def gen_txt(file='sampler.txt'):
    out = open(file, 'w')
    df = pd.read_csv("metadata.csv", index_col='image_id')
    for row in df.itertuples():
        a = row.Index
        valdf = df.drop(a, axis = 0)
        pdf = valdf[valdf.whale_id==row.whale_id]
        ndf = valdf[valdf.whale_id!=row.whale_id]
        if len(pdf)==0:
            print(f"Only one sample of this class: {a}")
            continue
        p = (pdf.sample(1)).index[0]
        n = (ndf.sample(1)).index[0]
        line = ' '.join([a, p, n])
        out.write("%s\n"% line)
    
    out.close()

if __name__ == "__main__":
    gen_txt()


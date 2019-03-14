import pandas as pd
from tqdm import tqdm

def get_V2_data():
    '''
    Requires Python 2.
    '''
    data = load_data.read_V2_data(None)[0]['sgRNA Score']
    genes_to_tests = {}
    for score, tup in zip(data, data.index):
        seq, gene, _ = tup
        if gene not in genes_to_tests:
            genes_to_tests[gene] = []
        genes_to_tests[gene].append((seq, score))
    return genes_to_tests

def get_150_exons_data():
    '''
    Get soft labeled Azimuth data.
    '''
    df = pd.read_csv('data/guide_options_top_1000_exons_w_preds.txt', delimiter='\t')
    genes = {}
    for i, gene, seq, pred in tqdm(list(df[['gene', 'guide_seq', 'azimuth_pred']].itertuples())):
        if gene not in genes:
            genes[gene] = []
        genes[gene].append((seq, pred))
    return genes
         
    

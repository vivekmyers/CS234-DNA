import Azimuth.azimuth.load_data as load_data

def get_V2_data():
    data = load_data.read_V2_data(None)[0]['sgRNA Score']
    genes_to_tests = {}
    for score, tup in zip(data, data.index):
        seq, gene, _ = tup
        if gene not in genes_to_tests:
            genes_to_tests[gene] = []
        else:
            genes_to_tests[gene].append((seq, score))
    return genes_to_tests

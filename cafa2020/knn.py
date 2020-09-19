

import os
import joblib
import sqlite3
import argparse
import percache
import functools
import numpy as np
from collections import Counter
import h5py
import timeit
import faiss
import pickle
from operator import itemgetter
import ntpath
from create_submission_files import *

COLS=1000
CONN = None
CURSOR = None
#DATA_DIR = '/home/user/CAFA/data/'
MODEL_PREFIX = '' #'embeddings_expanded'
DATA_DIR = '/home/user/Desktop/fast_storage/'
MODEL_FNAME = DATA_DIR + MODEL_PREFIX + '.sav'

CURSOR = CONN.cursor()
CONN = sqlite3.connect(DATA_DIR + 'protein_annotations.db')
MAX_ANNOTATIONS_PER_PROTEIN = 1500


ind2acc = {}

def train(fname, K=10000):
    global ind2acc
    index = faiss.IndexFlatL2(COLS)
    h5f = h5py.File(fname,'r')
    # X = h5f[list(h5f.keys())[0]][:]
    # print(f'Reading a matrix of size {X.shape[0]} by {X.shape[1]}')
    start_time = timeit.default_timer()
    for i,acc in enumerate(h5f.keys()):
        ind2acc[i] = acc
        # arr = h5f[list(h5f.keys())[0]][int(i*1e5):int((i+1)*1e5)].astype('float32')
        index.add(np.asmatrix(h5f.get(acc)[:]))
    print(f'Loaded {index.ntotal} vectors to index')
    h5f.close()
    elapsed = timeit.default_timer() - start_time
    print(f'Fitted in {elapsed}')
    # save the model to disk
    # joblib.dump(index, MODEL_FNAME)
    faiss.write_index(index, MODEL_FNAME)
    pickle.dump(ind2acc, open(DATA_DIR+MODEL_PREFIX + "train_ind2acc.pkl", "wb" ) )

def predict_and_annot(fname, K=10000):
    ind2acc = pickle.load( open(DATA_DIR+ MODEL_PREFIX + "train_ind2acc.pkl", "rb" ))
    index = faiss.read_index(MODEL_FNAME)
    print(f'Loaded {index.ntotal} vectors to index')
    h5f = h5py.File(fname, 'r')
    start_time = timeit.default_timer()
    for cafa_ind in h5f.keys():
        # ind2acc[i] = acc
        inds = single_pred(h5f.get(cafa_ind)[:], index, k=K)
        inds = [i for i in inds[0] if i > -1]
        annotation_predictions = accs_to_go_dicts([ind2acc[i] for i in inds])
        # Remove empty annotations
        if '' in annotation_predictions:
            annotation_predictions.pop('')
        # cafa_id = acc
        # for cafa_id, annotation_predictions in zip(batch_cafa_ids, batch_annotation_predictions):
        top_predicted_annotations = list(sorted(annotation_predictions.items(), key = itemgetter(1), reverse = True))[:MAX_ANNOTATIONS_PER_PROTEIN]
        top_predicted_annotations = [(annotation_index, round(score, 2)) for annotation_index, score in top_predicted_annotations]
        yield cafa_ind, top_predicted_annotations

GO_IND_TO_ID = {}
with open('/home/user/Desktop/fast_storage/go_annotations.csv') as f:
    for line in f:
        gid,ind = line.split(',')[:2]
        GO_IND_TO_ID[ind] = gid

def predict(fname, K=10000):
    h5f = h5py.File(fname, 'r')
    # X = h5f['dataset_1'][:]
    # Get first data set
    # X = h5f[list(h5f.keys())[0]][:]
    X = h5f[list(h5f.keys())[0]][:].astype('float32')
    h5f.close()
    # load the model from disk
    # loaded_model = joblib.load(MODEL_FNAME)
    index = faiss.read_index(MODEL_FNAME)
    # result = loaded_model.kneighbors(X, return_distance=False, n_neighbors=K)
    D, I = index.search(X, k)
    # Return indexes, ignore distances
    return I

def single_pred(x, index, k=10000):
    dists, inds = index.search(np.asmatrix(x), k)
    return inds

def print_neighbors(results, k):
    for i,vals in enumerate(results):
        print(str(i) + ' ' + ','.join(vals[:k]))

def accs_to_go_dicts(accs):
    gos = [i for a in accs for i in get_gos(a)]
    cntr = Counter(gos)
    return {g:i/len(accs) for g,i in cntr.items()}
    
# @functools chache within execution where @cache save to file for future execution as well
# @functools.lru_cache(maxsize=None)
cache = percache.Cache(DATA_DIR + "my-cache")
@cache
def get_gos(acc): 
    CURSOR.execute('SELECT complete_go_annotation_indices FROM protein_annotations WHERE uniprot_name="' + acc + '"')
    vals = CURSOR.fetchone()
    if vals is not None and len(vals) > 0:
        vals = vals[0].strip('][').split(', ')
        return vals
    return []

def print_annotations(results, k):
    global CONN
    global CURSOR
    CONN = sqlite3.connect(DATA_DIR + 'protein_annotations.db')
    CURSOR = CONN.cursor()
    for acc, neighbors in results:
        neighbors = neighbors[:k]
        gos = [i for a in neighbors for i in get_gos(a)]
        gos = [g for g in gos if g != ''] # Remove '' strings
        cntr = Counter(gos)
        print(acc + ' ' + ','.join([f'{g}:{i/n}' for g,i in cntr.items()]))

def main():
    parser = argparse.ArgumentParser(description='Train and run KNN')
    parser.add_argument('--train', help='Name of file with training vectors (hdf5)')
    parser.add_argument('--targets', default=None, help='Name of targets file')
    parser.add_argument('-k', '--k', default=1000, type=int, help='K (nearest neighbors)')
    parser.add_argument('-m', '--model_id', default=None, type=int, help='Model number for printing')
    parser.add_argument('--annotations', default=False, action='store_true', help='Print annotations, if False, print the neighbors ids. Default is False')
    args = parser.parse_args()
    if(args.train is not None):
        global MODEL_PREFIX
        MODEL_PREFIX = os.path.splitext(ntpath.basename(args.train))[0]
        train(args.train, args.k)
    if args.targets is not None and args.annotations:
        create_submission_files2(args.targets, model_id = args.model_id, K=args.k)
        
if __name__ == "__main__":
    main()

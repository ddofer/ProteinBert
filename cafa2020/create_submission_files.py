import os
import re
import json
from operator import itemgetter

import numpy as np
import pandas as pd

from knn import *

DATA_DIR = '/home/user/Desktop/fast_storage/'
target_seqs = pd.read_csv(DATA_DIR + 'target_seqs_expanded_annotations.csv.gz')

go_annotation_index_to_id = {}
with open('/home/user/Desktop/fast_storage/go_annotations.csv') as f:
    for line in f:
        gid,ind = line.split(',')[:2]
        go_annotation_index_to_id[ind] = gid

def process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, \
        batch_go_annotation_indices):
    
    MAX_ANNOTATIONS_PER_PROTEIN = 1500
    
    batch_annotation_predictions = prediction_model(batch_seqs, batch_go_annotation_indices)
    
    for cafa_id, annotation_predictions in zip(batch_cafa_ids, batch_annotation_predictions):
        top_predicted_annotations = list(sorted(annotation_predictions.items(), key = itemgetter(1), reverse = True))\
                [:MAX_ANNOTATIONS_PER_PROTEIN]
        top_predicted_annotations = [(annotation_index, round(score, 2)) for annotation_index, score in \
                top_predicted_annotations]
        formatted_predictions.extend(['%s\t%s\t%.2f' % (cafa_id, go_annotation_index_to_id[annotation_index], score) for \
                annotation_index, score in top_predicted_annotations if score > 0])

def create_submission_files(prediction_model, batch_size, output_dir, model_id = 1):
    
    '''
    prediction model is expected to be a function that takes two arguments (seqs and annotations) and returns the final,
    refined annotations. The function is expected to work in batch (i.e. receive multiple inputs and produce multiple 
    outputs). The protein seqs is expected as a list of strings of aa letters. The input annotations are expected
    as a list of list of integers, as provided in the SQLITE DB. There should be full correspondence between each seq
    string to each set of annotations (list of integers); each pair is considered a distinct protein. The output
    annotations is expected as a list of dictionaries, mapping each annotation integer (the same indices provided by
    the SQLITE DB) into a confidence score between 0 to 1. Each output dictionary corresponds to the corresponding input
    protein.
    
    See: https://www.biofunctionprediction.org/cafa-targets/CAFA4_rules%20_01_2020_v4.pdf
    '''
    # prediction_model receive a vector (embeded sequence) return a dict mapping from goID to score [0,1]
    # Make sure the submission files pass the git test from CAFA
    OUTPUT_FILE_NAME_PATTERN = 'linialgroup_%d_%s_go.txt'
    SUBMISSION_FILE_PREFIX = 'AUTHOR Linial' + '\n' + ('MODEL %d' % model_id) + '\n' + 'KEYWORDS machine learning.'
    
    for tax_id, tax_target_seqs in target_seqs.groupby('taxa_id'):
        
        print('Preparing submission for tax ID %s...' % tax_id)
        formatted_predictions = []
        batch_cafa_ids = []
        batch_seqs = []
        batch_go_annotation_indices = []
        
        for _, (cafa_id, seq, raw_go_annotation_indices) in tax_target_seqs[['cafa_id', 'seq', \
                'complete_go_annotation_indices']].iterrows():
            
            batch_cafa_ids.append(cafa_id)
            batch_seqs.append(seq)
            batch_go_annotation_indices.append([] if pd.isnull(raw_go_annotation_indices) else \
                    json.loads(raw_go_annotation_indices))
            
            if len(batch_cafa_ids) >= batch_size:
                process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, \
                        batch_go_annotation_indices)
                batch_cafa_ids = []
                batch_seqs = []
                batch_go_annotation_indices = []
                
        if len(batch_cafa_ids) > 0:
            process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, \
                    batch_go_annotation_indices)
            
        tax_submission_content = SUBMISSION_FILE_PREFIX + '\n' + '\n'.join(formatted_predictions) + '\n' + 'END'
        
        with open(os.path.join(output_dir, OUTPUT_FILE_NAME_PATTERN % (model_id, tax_id)), 'w') as f:
            f.write(tax_submission_content)
            
    print('Done.')
    
def create_submission_files_with_keras_model(keras_model, batch_size, output_dir, model_id = 1):
    
    def prediction_model(batch_seqs, batch_annotations):

        max_len = max(map(len, batch_seqs))
        batch_encoded_seqs = encode_seqs(batch_seqs, max_len = max_len)
        batch_encoded_annotations = encode_annotations(batch_annotations)
                
        _, batch_pred_annotation_scores = keras_model.predict(batch_encoded_seqs, batch_encoded_annotations)
        batch_annotation_predictions_as_dicts = []
        
        for (annotations, pred_annotation_scores) in zip(batch_annotations, batch_pred_annotation_scores):
            annotation_predictions_as_dicts = {unique_annotations[i]: score for i, score in \
                    enumerate(pred_annotation_scores)}
            annotation_predictions_as_dicts.update({annotation: 1.0 for annotation in annotations})
            batch_annotation_predictions_as_dicts.append(annotation_predictions_as_dicts)
            
        return batch_annotation_predictions_as_dicts
    
    create_submission_files(prediction_model, batch_size = batch_size, output_dir = output_dir, model_id = model_id)


def create_submission_files2(fname='/home/user/Desktop/fast_storage/target_embeddings2.hdf5', output_dir=DATA_DIR, model_id = 1, K=10000):
    
    '''
    prediction model is expected to be a function that takes two arguments (seqs and annotations) and returns the final,
    refined annotations. The function is expected to work in batch (i.e. receive multiple inputs and produce multiple 
    outputs). The protein seqs is expected as a list of strings of aa letters. The input annotations are expected
    as a list of list of integers, as provided in the SQLITE DB. There should be full correspondence between each seq
    string to each set of annotations (list of integers); each pair is considered a distinct protein. The output
    annotations is expected as a list of dictionaries, mapping each annotation integer (the same indices provided by
    the SQLITE DB) into a confidence score between 0 to 1. Each output dictionary corresponds to the corresponding input
    protein.
    
    See: https://www.biofunctionprediction.org/cafa-targets/CAFA4_rules%20_01_2020_v4.pdf
    '''
    # prediction_model receive a vector (embeded sequence) return a dict mapping from goID to score [0,1]
    # Make sure the submission files pass the git test from CAFA
    OUTPUT_FILE_NAME_PATTERN = 'linialgroup_%d_%s_go.txt'
    SUBMISSION_FILE_PREFIX = 'AUTHOR Linial' + '\n' + ('MODEL %d' % model_id) + '\n' + 'KEYWORDS machine learning.'
    tax_open = set()
    for cafa_ind, top_predicted_annotations in predict_and_annot(fname, K=K):

        #tax_id, tax_target_seqs in target_seqs.groupby('taxa_id'):
        taxa_id, cafa_id, uniprot_name, seq, raw_go_annotation_indices = target_seqs.loc[int(cafa_ind),:].values # target_seqs.loc[target_seqs['uniprot_name'] == acc].values[0]

        formatted_predictions = ['%s\t%s\t%.2f' % (cafa_id, GO_IND_TO_ID[annotation_index], score) for \
                annotation_index, score in top_predicted_annotations if score > 0]

        if taxa_id not in tax_open:
            tax_submission_content = SUBMISSION_FILE_PREFIX + '\n' + '\n'.join(formatted_predictions)
            tax_open.add(taxa_id)
        else:
            tax_submission_content = '\n'.join(formatted_predictions)
        
        with open(os.path.join(output_dir, OUTPUT_FILE_NAME_PATTERN % (model_id, taxa_id)), 'a') as f:
            f.write(tax_submission_content + '\n')
    # Close all file
    for taxa_id in tax_open:
        with open(os.path.join(output_dir, OUTPUT_FILE_NAME_PATTERN % (model_id, taxa_id)), 'a') as f:
            f.write('END\n')
    print('Done.')

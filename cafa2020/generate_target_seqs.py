MultiHeadAttention#
TARGET_SEQS_CSV_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/target_seqs_expanded_annotations.csv.gz'

def generate_target_seqs_batches(batch_size = 32, min_max_len = 1000):
    
    target_seqs = pd.read_csv(TARGET_SEQS_CSV_FILE_PATH)
    target_seqs['seq_len'] = target_seqs['seq'].str.len()
    target_seqs.sort_values('seq_len', inplace = True, ascending = False)
    
    while len(target_seqs) > 0:
        
        batch_seqs = target_seqs.iloc[:batch_size]
        target_seqs = target_seqs.iloc[batch_size:]
    
        batch_encoded_seqs = encode_seqs(batch_seqs['seq'], max_len = max(min_max_len, batch_seqs['seq_len'].max()))
        batch_encoded_annotations = encode_annotations(batch_seqs['complete_go_annotation_indices'])

        yield batch_seqs.index.values, batch_seqs['cafa_id'].values, batch_encoded_seqs, batch_encoded_annotations

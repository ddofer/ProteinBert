

import os
import re
import string
import urllib
import pandas as pd
import urllib.request
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from itertools import groupby
from tensorflow.keras import layers

def download_file(file_url: str, data_dir: str = "./data"):
    file_name = file_url.split("/")[-1]
    if not os.path.isfile(os.path.join(data_dir, file_name)):
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        print(f"Downloading {file_url}. Warning - slower than downloading externally")
        urllib.request.urlretrieve(file_url, os.path.join(data_dir, file_name))

def deduplicate_list(res: list):
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(res))

def extract_scop(file_path:str,split_char:str=" ", label_index:int=1, max_records = 1e6) -> list:
    record_iterator = SeqIO.parse(file_path, "fasta")
    all_records = []
    for seq_record in record_iterator:
        if len(all_records)<=max_records:
            protein_class = seq_record.description.split(split_char)[label_index] # this is 
            protein_sequence = str(seq_record.seq).upper()
            all_records.append((
                                protein_sequence,
                                protein_class
                               ))
    return(all_records)

def sliding_truncate_df_seqs_lengthwise(row,max_length:int = 4096):
    r_len = len(row)
    if r_len > max_length:
        return(row[0:max_length//2] + row[-max_length//2:]) # take first and last segments up to max length total
    return row

def cut_string(text:str,get_first_part=True)->str:
    midpoint = len(text)//2
    if get_first_part:
        return(text[0:midpoint])
    else:
        return(text[midpoint:])

def fasta_iter(fasta_name,max_records:int=1e8,seq_only:bool=True,MAX_LEN:int= None):
    fh = open(fasta_name)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for i,header in enumerate(faiter):
        if i>max_records: break
        header = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())
        if MAX_LEN!=None:
            seq = sliding_truncate_df_seqs_lengthwise(row=seq,max_length = MAX_LEN)
        if seq_only: 
            yield seq
        else:
            yield header, seq
            
def fasta_to_df(fasta_path,max_records:int=1e9,seq_only=False,MAX_LEN= None):
    return pd.DataFrame(fasta_iter(fasta_name=fasta_path,max_records=max_records,seq_only=seq_only,MAX_LEN= None))

def normalize_word(data_str):
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    data_str = data_str.lower()
    data_str = url_re.sub(' ', data_str)
    data_str = mention_re.sub(' ', data_str)
    data_str = punc_re.sub(' ', data_str)
    data_str = num_re.sub(' ', data_str)
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    return cleaned_str

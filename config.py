
import os
import model_cnn
import model_mpnn

# ----- Global -----

N_ANNOTATIONS = 8943
MODEL_TYPE = model_cnn
MAX_GLOBAL_SEQ_LEN = 450
ADDED_TOKENS_PER_SEQ = 2
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']


# ----- Pretraining -----

PRETRAINING_N = 10
PRETRAINING_SAVE_EVERY = 10
PRETRAINING_FIXED_SEED = True
PRETRAINING_RESUME_EPOCH = None
PRETRAINING_EPISODE_SETTINGS = [ (450, 32) ]


# ----- Fine-tuning -----

FINETUNING_MAX_EPOCHS = 80
FINETUNING_BATCH_SIZE = 8
FINETUNING_FAST_RUN = False
FINETUNING_DEBUG_MODE = False
FINETUNING_FAST_SAMPLE_RATIO = 0.03
FINETUNING_EARLY_STOPPING_PATIENCE = 4
FINETUNING_USE_PRETRAINED_WEIGHTS = True
FINETUNING_MAX_ALLOWED_INPUT_SEQ = MAX_GLOBAL_SEQ_LEN - 2

FINETUNING_BENCHMARKS = [
                            'signalP_binary.dataset',
                            # 'scop.dataset',
                            # 'remote_homology',
                            # 'secondary_structure',
                            # 'stability'
                            # 'phosphoserine.dataset',
                            # 'disorder_secondary_structure',
                            # 'PhosphositePTM.dataset',
                            # 'fluorescence',
                        ]

if FINETUNING_FAST_RUN:
    FINETUNING_MAX_EPOCHS = 1
    FINETUNING_EARLY_STOPPING_PATIENCE = 2
    FINETUNING_MAX_ALLOWED_INPUT_SEQ = 132
    FINETUNING_BATCH_SIZE = FINETUNING_BATCH_SIZE // 2


# ----- Setup specific (path's, weights, etc..) -----

if os.path.exists('../data'):

    # ----- Nadav's setup -----

        BENCHMARKS_DIR = '../data'
        H5_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/dataset.h5'
        PRETRAINING_BASE_WEIGHTS_DIR = '/cs/phd/nadavb/cafa_project/data/model_weights'
        FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH = '' # TODO Fill me up

    # ----- Nadav's setup -----

elif os.path.exists('/t/Dev/Yam'):

    # ----- Yam's setup -----

        BENCHMARKS_DIR = '/home/user/Desktop/slow_storage/protobert_new/data'
        H5_FILE_PATH = '/home/user/Desktop/slow_storage/protobert_new/dataset.h5'
        PRETRAINING_BASE_WEIGHTS_DIR = '/home/user/Desktop/slow_storage/protobert_new'
        FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH = '/home/user/Desktop/slow_storage/protobert_new/models/epoch_28530_sample_91400000.pkl'

    # ----- Yam's setup -----

else:
    # ----- Dan's setup -----

        BENCHMARKS_DIR = ''               # TODO Fill me up
        H5_FILE_PATH = ''                 # TODO Fill me up
        PRETRAINING_BASE_WEIGHTS_DIR = '' # TODO Fill me up
        FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH = '../models/fresh-paper/epoch_9840_sample_69400000.pkl'

    # ----- Dan's setup -----

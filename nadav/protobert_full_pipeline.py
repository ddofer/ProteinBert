import tensorflow as tf
from tensorflow import keras

from util import log, choose_from_cartesian_product, get_slurm_job_array_ids

from conv_and_global_attention_model import create_model
from pretraining import EpochGenerator, run_pretraining
from benchmark import ModelSpecification, run_benchmark

# TODO: Try different EpochGenerator parameters (e.g. p_seq_noise)
PARAM_OPTTIONS = {
    'model_d_hidden_seq': [128, 256, 512],
    'model_d_hidden_global': [512, 1024, 2048],
    'model_n_blocks': [1, 3, 6],
    'model_n_heads': [2, 8],
    'model_d_key': [32, 64, 128],
    'model_conv_kernel_size': [9],
    'model_wide_conv_dilation_rate': [5],
    'pretraining_n_epochs': [500],
    'pretraining_lr': [2e-04],
    'pretraining_annots_loss_weight': [1e03],
    'pretraining_fit_callbacks': [[]],
    'finetuning_benchmark': ['secondary_structure'],
    'finetuning_initial_lr': [1e-03],
    'finetuning_max_epochs': [30],
    'finetuning_batch_size': [8],
}

# Using smaller batch sizes in pretraining than the default episode settings, to allow running on GPUs with less memory.
PRETRAINING_EPISODE_SETTINGS = [
    # max_seq_len, batch_size
    (100, 32),
    (450, 16),
    (1200, 8),
]

if not tf.test.is_gpu_available():
    log('GPU is not available!')
    raise Exception()

param_options_keys, param_options_values = map(list, zip(*PARAM_OPTTIONS.items()))
job_id, total_tasks, task_index = get_slurm_job_array_ids()
chosen_param_values = choose_from_cartesian_product(param_options_values, task_index, total_tasks)
params = dict(zip(param_options_keys, chosen_param_values))
log('Params: %s' % params)

create_model_kwargs = dict(d_hidden_seq = params['model_d_hidden_seq'], d_hidden_global = params['model_d_hidden_global'], n_blocks = params['model_n_blocks'], \
        n_heads = params['model_n_heads'], d_key = params['model_d_key'], conv_kernel_size = params['model_conv_kernel_size'], \
        wide_conv_dilation_rate = params['model_wide_conv_dilation_rate'])

if params['pretraining_n_epochs'] == 0:
    model_weights = None
else:
    log('Pretraining for %d epochs...' % params['pretraining_n_epochs'])
    model_trainer = run_pretraining(create_model, EpochGenerator(episode_settings = PRETRAINING_EPISODE_SETTINGS), \
            n_epochs = params['pretraining_n_epochs'], lr = params['pretraining_lr'], annots_loss_weight = params['pretraining_annots_loss_weight'], \
            fit_callbacks = params['pretraining_fit_callbacks'], model_creation_kwargs = create_model_kwargs)
    model_weights = model_trainer.model.get_weights()

log('Fine-tuning on the %s benchmark for maximum %d epochs...' % (params['finetuning_benchmark'], params['finetuning_max_epochs']))
model_specification = ModelSpecification(create_model, model_weights = model_weights, create_model_kwargs = create_model_kwargs)
run_benchmark(params['finetuning_benchmark'], model_specification, settings = dict(initial_lr = params['finetuning_initial_lr'], \
        max_epochs = params['finetuning_max_epochs'], batch_size = params['finetuning_batch_size']), debug = False)
        
log('Done.')
import pickle

import tensorflow.keras.backend as K

def get_model_creator(n_annotations, create_model_function, **kwargs):

    def model_creator(max_seq_len):
        from tokenization import n_tokens
        K.clear_session()
        return create_model_function(max_seq_len, n_tokens, n_annotations, **kwargs)
        
    return model_creator

def recreate_model_with_same_state(old_model, create_and_compile_model_funtion):
    model_weights, optimizer_weights = old_model.get_weights(), old_model.optimizer.get_weights()
    new_model = create_and_compile_model_funtion()
    new_model.set_weights(model_weights)
    new_model.optimizer.set_weights(optimizer_weights)
    return new_model

def save_model_state(model, path):
    with open(path, 'wb') as f:
        pickle.dump((model.get_weights(), model.optimizer.get_weights()), f)
        
def load_model_state(model, path):
    with open(path, 'rb') as f:
        model_weights, optimizer_weights = pickle.load(f)
        model.set_weights(model_weights)
        model.optimizer.set_weights(optimizer_weights)
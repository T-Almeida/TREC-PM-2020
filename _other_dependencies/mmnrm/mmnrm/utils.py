import numpy as np
import random
import tensorflow as tf
import h5py
import pickle
import mmnrm.modelsv2
from datetime import datetime as dt

def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    
def save_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'w') as f:
        weight = model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight'+str(i), data=weight[i])

def load_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)
        
def save_model(file_name, model):
    cfg = model.savable_config
    with open(file_name+".cfg","wb") as f:
        pickle.dump(model.savable_config ,f)
        
    # keep using h5py for weights
    save_model_weights(file_name, model)
    
def load_model(file_name, change_config={}):
    
    with open(file_name+".cfg","rb") as f:
        cfg = pickle.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    # create the model with the correct configuration
    model = getattr(mmnrm.modelsv2, cfg['func_name'])(**cfg)
    
    # load weights
    load_model_weights(file_name, model)
    
    return model
        
def merge_dicts(*list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp

def flat_list(x):
    return sum(x, [])

def index_from_list(searchable_list, comparison):
    for i,item in enumerate(searchable_list):
        if comparison(item):
            return i
    return -1

def overlap(snippetA, snippetB):
    """
    snippetA: goldSTD
    """
    if snippetA[0]>snippetB[1] or snippetA[1] < snippetB[0]:
        return 0
    else:
        if snippetA[0]>=snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetA[0] + 1
        if snippetA[0]>=snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetB[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        
    return 0

def to_date(_str):
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y"):
        try:
            return dt.strptime(_str, fmt)
        except ValueError:
            pass
    raise ValueError("No format found")

# %%
import torch, sys
sys.path.insert(0, '../')
from my_utils import gpu_utils
import importlib, gc
from my_utils.alignment_features import *
import my_utils.alignment_features as afeatures
importlib.reload(afeatures)
import postag_utils as posutil
from my_utils.pytorch_utils import EarlyStopping
import os



# %%
import random
from gensim.models import Word2Vec
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import torch

import torch_geometric.transforms as T


# %%
from my_utils import align_utils as autils, utils
import argparse
from multiprocessing import Pool
import random

path_ext = '/inter'
gdfa = False
adar = False
small = True
if gdfa:
    path_ext = '/gdfa_final'
elif adar:
    if small:
        path_ext = '/small_adar_final'
    else:
        path_ext = '/adar_final'
elif small:
    path_ext = '/small_final'

# set random seed
config_file = "/mounts/Users/student/ayyoob/Dokumente/code/POS-TAGGING/config_pos.ini"
utils.setup(config_file)

params = argparse.Namespace()


params.editions_file =  "/mounts/Users/student/ayyoob/Dokumente/code/POS-TAGGING/editions_listsmall2.txt"
utils.graph_dataset_path =  '/mounts/data/proj/ayyoob/POS_tagging/dataset/' 
editions, langs = autils.load_simalign_editions(params.editions_file)
current_editions = []
for lang in langs:
    if editions[lang] not in current_editions:
        current_editions.append(editions[lang])

# %%

xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()
w2v_model = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_POS_small_final_langs_10e.model")
train_dataset = torch.load(utils.graph_dataset_path +  f'{path_ext}/dataset_forpos1_word.torch.bin')
#train_dataset.__dict__['_store'] = False # this is to fix version incompatibility issues

# %%
import tqdm

def get_xlmr_embeddings(nodes_map, verse, languages, w2v_model, nodes_count, xlmr, save_path, x):
    to_save = {}
    if os.path.exists(f'{save_path}/xlmr_features3/{verse}.torch.bin'):
        #to_save = torch.load(f'{save_path}/xlmr_features3/{verse}.torch.bin')
        #return to_save['res']
        return None

    res = torch.zeros(nodes_count, 1024)
    updates = set()
    sorted_sentences = {}
    for l in languages:
        if l in nodes_map and verse in nodes_map[l]:
            items = nodes_map[l][verse].items()
            sorted_items = [-1 for i in range(len(items))]
            for item in items:
                sorted_items[item[0]] = item[1]
            sorted_sentences[l] = sorted_items

            xlm_token_ids = [torch.tensor(0)]
            tid_to_xlmid = {}
            for tok_id in sorted_items:
                token = w2v_model.wv.index_to_key[x[tok_id].item()][4:] # TODO fixme eng:apple
                xlm_tis = xlmr.encode(token)[1:-1]
                tid_to_xlmid[tok_id] = [len(xlm_token_ids)]
                xlm_token_ids.extend(xlm_tis)
                if len(xlm_tis) > 1:
                    tid_to_xlmid[tok_id].append(len(xlm_token_ids) -1)
            xlm_token_ids.append(torch.tensor(2))

            all_xlm_features = xlmr.extract_features(torch.tensor(xlm_token_ids))
            
            for tok_id in sorted_items:
                selecte_xlm_features = all_xlm_features[0, tid_to_xlmid[tok_id], :]
                
                selecte_xlm_features = torch.sum(selecte_xlm_features, dim=0)
                res[tok_id] = selecte_xlm_features
            for xx in sorted_items:
                updates.add(xx)

    to_save['res'] = res
    to_save['sorted_sentences'] = sorted_sentences
    assert len(updates) == nodes_count
    torch.save(to_save, f'{save_path}/xlmr_features3/{verse}.torch.bin')

    return res

begin = int(sys.argv[1])

end = 300000
print('range', begin, end)
jump = 0
res = 1
for verse in tqdm.tqdm(train_dataset.accepted_verses[begin:]):
    if res == None and jump < 500:
        jump += 1
        continue
    res = 1
    jump = 0
    verse_info = torch.load(utils.graph_dataset_path + path_ext + f'/verses/{verse}_info.torch.bin')
    res = get_xlmr_embeddings(train_dataset.nodes_map, verse, current_editions, w2v_model, verse_info['x'].size(0), xlmr,  utils.graph_dataset_path + path_ext + '/', verse_info['x'][:,9].long())
    


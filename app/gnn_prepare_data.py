# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch, sys
sys.path.insert(0, '../')
import importlib
import gc
from my_utils import align_utils as autils, utils
from my_utils.alignment_features import *
from tqdm import tqdm
import torch

def remove_bad_editions(editions_list):
    bad_editions = []

    for edition in editions_list:
        total_tokens = 0
        total_lines = 0
        try:
            with open(f'/nfs/datc/pbc/{edition}.txt') as fi:
                for i,line in enumerate(fi):
                    if i>10:
                        total_tokens += len(list(line.split()))
                        total_lines += 1
            
            if total_tokens/total_lines<10:
                bad_editions.append[edition]
                #print(edition, total_tokens, total_lines)
        except:
            print('couldn\'t open edition', edition)

    if len(bad_editions) > 0:
        print('!!!WARNING!!!     The following files are not tokenized properly. deleting them...')
        print(bad_editions)
        for edition in bad_editions:
            editions_list.remove(edition)
    
def get_verse_list(file):
    res = []
    with open(file, 'r') as inf:
        for line in inf:
            res.append(line.strip())
    return res

# %%
import argparse

config_file = "/mounts/Users/student/ayyoob/Dokumente/code/POS-TAGGING/config_pos.ini"
utils.setup(config_file)

params = argparse.Namespace()
params.editions_file =  "/mounts/Users/student/ayyoob/Dokumente/code/POS-TAGGING/editions_listsmall2.txt"
utils.graph_dataset_path =  '/mounts/data/proj/ayyoob/POS_tagging/dataset/'

editions, langs = autils.load_simalign_editions(params.editions_file)
current_editions = [editions[lang] for lang in langs]
remove_bad_editions(current_editions)

all_verses = get_verse_list(utils.config_dir + f"/verse_list.txt")


# %%
import gnn_utils.graph_utils as gutil
importlib.reload(gutil)
sys.setrecursionlimit(100000)

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

#train_dataset, train_nodes_map = gutil.create_dataset(all_verses, gdfa, adar, current_editions, utils.graph_dataset_path + path_ext)

#torch.save(train_dataset, utils.graph_dataset_path + f"{path_ext}/dataset_forpos.torch.bin")
#train_dataset = torch.load(utils.graph_dataset_path + f"{path_ext}/dataset_forpos.torch.bin")
#features = train_dataset.features
#train_nodes_map = train_dataset.nodes_map

#exit(0)
# %%
# This cell is to fix the dataset object if it has been created by multiple calls to create_dataset function and some community lengths and features are missing
#import os

#failed_tocreate_verses = []
#for verse in tqdm(train_dataset.max_sentece_sizes.keys()):
#    if verse not in train_dataset.community_lens and os.path.exists(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin"):
#        verse_info = torch.load(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
#        train_dataset.community_lens[verse] = torch.max(torch.tensor(verse_info['x'])[:, 7:])
#    elif verse in train_dataset.community_lens:
#        pass
#    else:
#        failed_tocreate_verses.append(verse)

#train_dataset.failed_tocreate_verses = failed_tocreate_verses
#print('failed to create: ', len(failed_tocreate_verses))

#torch.save(train_dataset, utils.graph_dataset_path + f"{path_ext}/dataset_forpos1_new.torch.bin")


#exit(0)
# %%
# to set proper verse_len and community_len in the corresponding features and filter verses with bigger values

#from pprint import pprint
#for feat in train_dataset.features:
#    print(vars(feat))

#max_community_len = 255
#max_verse_len = 255

#refused_verses = set()
#accepted_verses = set()
#def set_max_sentence_and_community(train_dataset):


#    for verse, sl in train_dataset.max_sentece_sizes.items():
#        if sl >max_community_len:
#            refused_verses.add(verse)


#    for verse,cl in train_dataset.community_lens.items():
#        if cl > max_community_len:
#            refused_verses.add(verse)

#    for verse in train_dataset.community_lens.keys():
#        if verse not in refused_verses:
#            accepted_verses.add(verse)

#    max_sentence = max(train_dataset.max_sentece_sizes.values())
#    max_community = max(train_dataset.community_lens.values())
#    print('dataset\'s max sentence len', max_sentence)
#    print('dataset\'s max community len: ', max_community)


#set_max_sentence_and_community(train_dataset)
#print(refused_verses)
#print('len refused verses:', len(refused_verses))
#print('verse count', len(train_dataset.community_lens))
#print('accepted verse count: ', len(accepted_verses))

#train_dataset.refused_verses = list(refused_verses)
#train_dataset.accepted_verses = list(accepted_verses)
#train_dataset.features[1].n_classes = max_verse_len + 1
#train_dataset.features[7].n_classes = max_community_len + 1
#train_dataset.features[8].n_classes = max_community_len + 1
#torch.save(train_dataset, utils.graph_dataset_path + f"{path_ext}/dataset_forpos_fixlens.torch.bin")
#exit(0)
train_dataset = torch.load(utils.graph_dataset_path + f"{path_ext}/dataset_forpos_fixlens.torch.bin")


    

# %%
#run on delta, extract w2v features
sys.path.insert(0, '../')
import pickle, os
from gensim.models import Word2Vec
from app import document_retrieval
import my_utils.alignment_features as feat_utils
importlib.reload(document_retrieval)
from random import randint

doc_retriever = document_retrieval.DocumentRetriever()

model_w2v = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_POS_small_final_langs_10e.model")
nodes_map = train_dataset.nodes_map

non_existing_verses = []
for verse in tqdm(train_dataset.verse_lengthes.keys()):
    if (os.path.exists(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
        and os.path.getsize(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin") > 0):
        
        verse_info = torch.load( utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
        
        x = verse_info['x']
        
        if not torch.is_tensor(x) or x.size(1) < 10:
            if torch.is_tensor(x):
                x = x.tolist()
            for edition_f in nodes_map:
                if verse in nodes_map[edition_f]:
                    line = doc_retriever.retrieve_document(f'{verse}@{edition_f}')
                    line = line.strip().split()

                    for tok in nodes_map[edition_f][verse]:
                        w_emb = model_w2v.wv.key_to_index[f'{edition_f[:3]}:{line[tok]}']
                        x[nodes_map[edition_f][verse][tok]].extend([w_emb])

            
            verse_info['x'] = torch.tensor(x, dtype=torch.float)
            torch.save(verse_info, utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
    else:
        non_existing_verses.append(verse)
    
# train_dataset.features.pop()
train_dataset.features.append(feat_utils.MappingFeature(100, 'word'))

print(verse_info['x'].shape, len(train_dataset.features))
if len(non_existing_verses) > 0:
    print('!!!WARNING!!! very bad news, these verses not found: ', non_existing_verses)

torch.save(train_dataset, utils.graph_dataset_path +  f'{path_ext}/dataset_forpos1_word.torch.bin') #europarl
print('done adding w2v features')

# %%
features = train_dataset.features[:]
from pprint import pprint


# %%
nodes_map = train_dataset.nodes_map
bad_edition_files = []
for edit in nodes_map:
    bad_count = 0
    for verse in nodes_map[edit]:
        if len(nodes_map[edit][verse].keys()) < 6:
            bad_count += 1
        if bad_count > 1:
            bad_edition_files.append(edit)
            break
print(bad_edition_files)


## %%
#all_japanese_nodes = set()
#nodes_map = train_dataset.nodes_map

#for verse in nodes_map['jpn-x-bible-newworld']:
#    for item in nodes_map['jpn-x-bible-newworld'][verse].items():
#        all_japanese_nodes.add(item[1])

#print(" all japansese nodes: ", len(all_japanese_nodes))
#edge_index = train_dataset.edge_index.to('cpu')
#remaining_edges_index = []
#for i in tqdm(range(0, edge_index.shape[1], 2)):
#    if edge_index[0, i].item() not in all_japanese_nodes and edge_index[0, i+1].item() not in all_japanese_nodes:
#        remaining_edges_index.extend([i, i+1])

#print('original total edges count', edge_index.shape)
#print('remaining edge count', len(remaining_edges_index))
#train_dataset.edge_index = edge_index[:, remaining_edges_index]
#train_dataset.edge_index.shape


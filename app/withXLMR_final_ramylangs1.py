# %%
import torch, sys, os
sys.path.insert(0, '../')
from my_utils import gpu_utils
import importlib, gc
from my_utils.alignment_features import *
import my_utils.alignment_features as afeatures
importlib.reload(afeatures)
import gnn_utils.graph_utils as gutils
import postag_utils as posutil
from my_utils.pytorch_utils import EarlyStopping

# %%
dev = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dev2 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

#from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt




# %%
from my_utils import align_utils as autils, utils
import argparse
from multiprocessing import Pool
import random

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
importlib.reload(afeatures)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features, n_head = 2, has_tagfreq_feature=False,):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, 2*out_channels, heads= n_head)
        self.conv2 = pyg_nn.GATConv(2 * n_head *  out_channels , out_channels, heads= n_head)
        #self.fin_lin = nn.Linear(out_channels, out_channels)
        
        if has_tagfreq_feature:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies, word_vectors])
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies])
            #self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies,train_pos_labels, word_vectors])
        else:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [word_vectors])
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [])
            #self.feature_encoder = afeatures.FeatureEncoding(features, [])
            #self.feature_encoder = afeatures.FeatureEncoding(features, [train_pos_labels, word_vectors])

    def forward(self, x, edge_index):
        encoded = self.feature_encoder(x, dev)
        
        x = F.elu(self.conv1(encoded, edge_index, ))
        x = F.elu(self.conv2(x, edge_index))
        #return F.relu(self.fin_lin(x)), encoded
        return x, encoded

class Encoder1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features, n_head = 2, has_tagfreq_feature=False,):
        super(Encoder1, self).__init__()
        #self.conv1 = pyg_nn.GATConv(in_channels, 2*out_channels, heads= n_head)
        self.fin_lin = nn.Linear(in_channels, out_channels)
        
        if has_tagfreq_feature:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies, word_vectors])
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies])
            #self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies,train_pos_labels, word_vectors])
        else:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [word_vectors])
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [])

    def forward(self, x, edge_index):
        encoded = self.feature_encoder(x, dev)
        
        return F.relu(self.fin_lin(encoded)), encoded

# %%
def clean_memory():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

class DataEncoder():

    def __init__(self, data_loader, model, mask_language):
        self.data_loader = data_loader
        self.model = model
        self.mask_language = mask_language
    
    def __iter__(self):
        for i,batch in enumerate(tqdm(self.data_loader)):
            x = batch['x'][0].to(dev)  # initial features (not encoded)
            edge_index = batch['edge_index'][0].to(dev) 
            #print(edge_index.shape)
            verse = batch['verse'][0]

            #index = batch['pos_index'][0].to(dev)
            #print('x', x[index, 10:28])
            # if verse in masked_verses:
            #     continue

            try:
                if self.mask_language:
                    x[:, 0] = 0
                z, encoded = self.model.encode(x, edge_index) # Z will be the output of the GNN
                batch['encoded'] = encoded
            except Exception as e:
                global sag, khar, gav
                sag, khar, gav =  (i, batch, verse)
                print(e)
                1/0
            
            yield z, verse, i, batch

def train(epoch, data_loader, mask_language, test_data_loader, max_batches=999999999):
    global optimizer
    total_loss = 0
    model.train()
    loss_multi_round = 0

    data_encoder = DataEncoder(data_loader, model, mask_language)
    optimizer.zero_grad()

    for z, verse, i, batch in data_encoder:

        target = batch['pos_classes'][0].to(dev)
        #print('labels', target)
        _, labels = torch.max(target, 1)
        
        index = batch['pos_index'][0].to(dev)

        preds = model.decoder(z, index, batch)
        
        loss = criterion(preds, labels)
        loss = loss * target.shape[0] # TODO check if this is necessary
        loss.backward()
        total_loss += loss.item()

        if (i+1) % 50 == 0: # Gradient accumulation
            optimizer.step()
            optimizer.zero_grad()
            #clean_memory() 
        
            

        if i % 1000 == 999:
            # print(f"loss: {total_loss}")
            total_loss = 0
            val_loss = test(epoch, test_data_loader, mask_language)
            # test_mostfreq(yor_data_loader, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(tam_data_loader, True, tam_gold_mostfreq_tag, tam_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(arb_data_loader, True, arb_gold_mostfreq_tag, arb_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(por_data_loader, True, por_gold_mostfreq_tag, por_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            print('----------------------------------------------------------------------------------------------------------')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            model.train()
            clean_memory()

        if i == max_batches:
            break
        
    print(f"total train loss: {total_loss}")


# %%
class POSDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_class, skip_connection, drop_out=0):
        super(POSDecoder, self).__init__()
        self.skip_connection = skip_connection
        self.transfer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(drop_out),
                        nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(drop_out),
                        nn.Linear(hidden_size, n_class))

    def forward(self, z, index, batch_=None):
        h = z[index, :]
        
        x = batch_['encoded'][index, :]
        if self.skip_connection:
            h = torch.cat((h,x), dim=1)

        res = self.transfer(h)

        return res

class POSDecoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, n_class, residual_connection, sequence_size, drop_out=0):
        super(POSDecoderTransformer, self).__init__()
        self.sequence_size = sequence_size
        self.residual_connection = residual_connection

        # for skip connection:
        self.input_size = input_size
        n_head = int(input_size/64)
        if n_head * 64 != input_size:
            n_head += 1
            self.input_size = n_head * 64
        

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=int(self.input_size/64), dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.transfer = nn.Sequential( nn.Linear(self.input_size, self.input_size*2), nn.ReLU(), nn.Dropout(drop_out), # TODO check what happens if I remove this.
                        nn.Linear(self.input_size*2, n_class))

    def forward(self, z_, index, batch_):
        z = z_.to(dev2)

        x = F.pad(batch_['encoded'], (0, self.input_size - z.size(1) - batch_['encoded'].size(1))).to(dev2)


        language_based_nodes = batch_['lang_based_nodes'] # determines which node belongs to which language
        transformer_indices = batch_['transformer_indices'] # the reverse of the prev structure

        sentences = []
        for lang_nodes in language_based_nodes: # we rearrange the nodes into sentences of each language
            if self.residual_connection:
                tensor = torch.cat((z[lang_nodes, :], x[lang_nodes, :]), dim=1)
            else:
                tensor = z[lang_nodes, :]
                
            try:
                tensor = F.pad(tensor, (0, 0, 0, self.sequence_size - tensor.size(0)))
            except Exception as e:
                print(self.sequence_size, tensor.size(0))
            sentences.append(tensor)
        
        batch = torch.stack(sentences) # A batch contains all translations of one sentence in all training languages.
        batch = torch.transpose(batch, 0, 1)

        h = self.transformer(batch)
        h = torch.transpose(h, 0, 1)
        h = h[transformer_indices[0], transformer_indices[1], :] # rearrange the nodes back to the order in which we recieved (the order that represents the graph)

        res = self.transfer(h)

        return res.to(dev)

# %%
torch.set_printoptions(edgeitems=2)

def test_goozooo(epoch, testloader, mask_language, filter_wordtypes=None):
    print('testing',  epoch)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    probability_sum = 0 
    probability_count = 0

    #data_encoder = DataEncoder(testloader, model, mask_language)
    #{'verse':verse, 'x':self.verse_info[verse]['x'], 'edge_index': edge_index if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long), 
    #            'pos_classes': self.pos_labels[verse][nodes, :], 'pos_index': torch.LongTensor(nodes) - padding, 
    #            'padding': padding, 'lang_based_nodes': language_based_nodes, 'transformer_indices': transformer_indices}
    with torch.no_grad():
        for i,batch in enumerate(tqdm(testloader)):
            #print(batch)
            target = batch['pos_classes'][0].to(dev)
            index = batch['pos_index'][0].to(dev)
            verse = batch['verse'][0]
            #print('verse', batch['verse'] , '\n')
            #print(target.shape, '\n')
            #print(batch['x'].shape, '\n')
            #print(verse, '\n')
            preds = train_pos_labels_ext[verse][index, :].to(dev)
            
            #print(preds.shape, '\n')
           

            if preds.size(0) > 0:
                max_probs, predicted = torch.max(preds, 1)
                #print(max_probs)
                filter = max_probs>0
                max_probs = max_probs[filter]
                predicted = predicted[filter]
                _, labels = torch.max(target, 1)
                labels = labels[filter]
                loss = 0
                probability_sum += torch.sum(max_probs)
                probability_count += max_probs.size(0)
                total_loss += loss
                
                for ii in range(predicted.size(0)):
                    silver_tags[labels[ii]] +=1
                    bronze_tags[predicted[ii]] +=1

            verse_correct = (predicted == labels).sum().item()
            total += labels.size(0)
            correct += verse_correct
            verse_accuracy = verse_correct/(labels.size(0)+0.00001)
            add_accuracy_to_bucket(verse_accuracy, verse)
            if i == 200:
                break

    print(f'test, epoch: {epoch}, total:{total} ACC: {correct/total}, loss: {total_loss}, confidence: {probability_sum/probability_count}')
    clean_memory()
    return 1.0 - correct/total

#silver_tags = [0 for i in range(17)]
#bronze_tags = [0 for i in range(17)]
#test_goozooo(0, val_data_loader_pos, False)
#print(bronze_tags, '\n\n')
#print(silver_tags)

# %%
import collections
verse_accuracy_buckets = collections.defaultdict(list)
verse_accuracies = {}

def add_accuracy_to_bucket(accuracy, verse):
    bottom = int(accuracy/5) * 5
    top = (int(accuracy/5) + 1) * 5

    bucket = f'{bottom}-{top}'
    verse_accuracies[verse] = accuracy
    verse_accuracy_buckets[bucket].append(verse)

def test(epoch, testloader, mask_language):
    print('testing',  epoch)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    probability_sum = 0 
    probability_count = 0

    data_encoder = DataEncoder(testloader, model, mask_language)
    
    with torch.no_grad():
        for z, verse, i, batch in data_encoder:
            
            target = batch['pos_classes'][0].to(dev)
            index = batch['pos_index'][0].to(dev)
            


            #print(target.shape, index.shape)
            preds = model.decoder(z, index, batch)
            #print(preds.shape)

            if preds.size(0) > 0:
                _, labels = torch.max(target, 1)
                
                actual_labels = labels != postag_map['X']
                labels = labels[actual_labels]
                preds = preds[actual_labels]

                max_probs, predicted = torch.max(torch.softmax(preds, dim=1), 1)
                

                loss = criterion(preds, labels)
                probability_sum += torch.sum(max_probs)
                probability_count += max_probs.size(0)
                #total_loss += loss

            verse_correct = (predicted == labels).sum().item()
            total += labels.size(0)
            correct += verse_correct
            verse_accuracy = verse_correct/(labels.size(0)+0.00001)
            add_accuracy_to_bucket(verse_accuracy, verse)
    print(f'test, epoch: {epoch}, total:{total} ACC: {correct/total}, loss: {total_loss}, confidence: {probability_sum/probability_count}')
    clean_memory()
    return 1.0 - correct/total

def test_mostfreq(testloader, mask_language, target_mostfreq_tags, target_mostfreq_index, word_types_shape, from_target=False):
    
    res = torch.zeros(word_types_shape[0], word_types_shape[1])
    model.eval()

    data_encoder = DataEncoder(testloader, model, mask_language)
    
    with torch.no_grad():
        for z, verse, i, batch in data_encoder:
            
            index = batch['pos_index'][0].to(dev)
            x = batch['x'][0][index, :]
            target = batch['pos_classes'][0]

            preds = model.decoder(z, index, batch)

            _, pred_max = torch.max(preds, dim=1)
            _, targe_max = torch.max(target, dim=1)

            if from_target:
                res[x[:, 9].long(), targe_max.long()] += 1
            else:
                res[x[:, 9].long(), pred_max.long()] += 1
    
    max_vals, res_tags = torch.max(res, dim=1)
    res_tags = res_tags[target_mostfreq_index]
    max_vals = max_vals[target_mostfreq_index]
    

    
    #print(f'correct = {torch.sum(target_mostfreq_tags == res_tags)}')
    #print(f'most frequency test, total:{target_mostfreq_tags.shape[0]}, accuracy:{torch.sum(target_mostfreq_tags == res_tags)/target_mostfreq_tags.shape[0]}, ')
    #print('target mostfreq tags', target_mostfreq_tags.shape)
    #print('res_tags', res_tags.shape)

    res_tags = res_tags[max_vals>0.1]
    target_mostfreq_tags_cp = target_mostfreq_tags[max_vals>0.1]
    #print(f'correct = {torch.sum(target_mostfreq_tags_cp == res_tags)}')
    print(f'most frequency test, total:{target_mostfreq_tags_cp.shape[0]}, accuracy:{torch.sum(target_mostfreq_tags_cp == res_tags)/target_mostfreq_tags_cp.shape[0]}, ')
    #print('target mostfreq tags', target_mostfreq_tags_cp.shape)
    #print('res_tags', res_tags.shape)


# %%
def majority_voting_test(data_loader1, data_loader2):
    total = 0
    correct = 0
    
    for i,(batch, batch2) in enumerate(tqdm(zip(data_loader1, data_loader2))) :
            
        x = batch['x'][0]
        edge_index = batch['edge_index'][0]
        verse = batch['verse'][0]

        # if verse in masked_verses:
        #     continue

        target = batch['pos_classes'][0]
        index = batch['pos_index'][0]

        index2 = batch2['pos_index'][0]
        

        for node, label in zip(index,target):
            other_side = edge_index[1, edge_index[0, :] == node]
            other_side_withpos = other_side[[True if i in index2 else False for i in other_side]]
            other_side_target_indices = [(i == index2).nonzero(as_tuple=True)[0].item() for i in other_side_withpos]
            #print(other_side_target_indices)
            proj_tags = batch2['pos_classes'][0][other_side_target_indices]

            if proj_tags.size(0) > 0:
                _, proj_tags = torch.max(proj_tags, 1)
                #print(target.shape, node, index.shape, proj_tags, other_side)
                
                if torch.argmax(label) == torch.mode(proj_tags)[0]:
                    correct += 1
                
                total += 1

    print(f'test, , total:{total} ACC: {correct/total}')


# %%
from gensim.models import Word2Vec

w2v_model = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_POS_small_final_langs_10e.model")


print(w2v_model.wv.vectors.shape)
word_vectors = torch.from_numpy(w2v_model.wv.vectors).float()

# %%
import pickle

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
    
train_dataset = torch.load(utils.graph_dataset_path +  f'{path_ext}/dataset_forpos1_word.torch.bin')
#train_dataset.__dict__['_store'] = False # this is to fix version incompatibility issues

original_verses = train_dataset.accepted_verses[:]

# %%

verse_sizes = {}
def get_verse_size_distrib(nodes_map, verses_filter):
    distrib = [0 for i in range(100)]
    for edition in nodes_map:
        for verse in nodes_map[edition]:
            if verses_filter == [] or verse in verses_filter:
                if verse in verse_sizes:
                    verse_sizes[verse] += 1
                else:
                    verse_sizes[verse] = 1
    
    for verse in verse_sizes:
        distrib[verse_sizes[verse]] += 1
    
    for i in range(len(distrib)):
        if distrib[i] > 10:
            print(i, distrib[i])

def update_verse_sizes(nodes_map, verse_sizes):
    for edition in nodes_map:
        for verse in nodes_map[edition]:
            if verse in verse_sizes:
                verse_sizes[verse] += 1
            else:
                verse_sizes[verse] = 1

def update_distribution(nodes_map, verse_sizes, distrib, edition):
    if edition in nodes_map:
        for verse in nodes_map[edition]:
            if verse_sizes[verse] > 99:
                print(verse, verse_sizes[verse])
                continue
            distrib[verse_sizes[verse]] += 1
                
def get_language_verse_distrib(nodes_map1, nodes_map2, nodes_map3, nodes_map4, edition):
    global all_verse_sizes
    all_verse_sizes = {}

    update_verse_sizes(nodes_map1, all_verse_sizes)
    update_verse_sizes(nodes_map2, all_verse_sizes)
    update_verse_sizes(nodes_map3, all_verse_sizes)
    update_verse_sizes(nodes_map4, all_verse_sizes)

    
    distrib = [0 for i in range(100)]
    update_distribution(nodes_map1, all_verse_sizes, distrib, edition)
    update_distribution(nodes_map2, all_verse_sizes, distrib, edition)
    update_distribution(nodes_map3, all_verse_sizes, distrib, edition)
    update_distribution(nodes_map4, all_verse_sizes, distrib, edition)
    
    for i in range(len(distrib)):
        if distrib[i] > 10:
            print(i, distrib[i])

# get_language_verse_distrib(train_dataset.nodes_map, blinker_test_dataset.nodes_map, grc_test_dataset.nodes_map, heb_test_dataset.nodes_map, 'tam-x-bible-newworld')
get_verse_size_distrib(train_dataset.nodes_map, [])
# get_verse_size_distrib(blinker_test_dataset.nodes_map)
# get_verse_size_distrib(grc_test_dataset.nodes_map)
# get_verse_size_distrib(heb_test_dataset.nodes_map)
# big_verses = []
# for verse in train_verses:
#     if all_verse_sizes[verse] > 50:
#         big_verses.append(verse)
# gnn_dataset_train_pos = POSTAGGNNDataset(train_dataset, big_verses, editions, {}, train_pos_node_cover, train_pos_labels, data_dir_train, group_size=128)
#train_dataset.accepted_verses = original_verses[:]
#print(len(train_dataset.accepted_verses))
#for verse in verse_sizes:
#    if verse_sizes[verse] < 61:
#        if verse in train_dataset.accepted_verses:
#            train_dataset.accepted_verses.remove(verse)
#print(len(train_dataset.accepted_verses))
shuffled_verses = train_dataset.accepted_verses[:]
random.shuffle(shuffled_verses)

new_testament_verses = []
old_testament_verses = []

starts = set()
for verse in train_dataset.accepted_verses:
    starts.add(verse[0])
    if verse[0] in ['4','5','6']:
        new_testament_verses.append(verse)
    else:
        old_testament_verses.append(verse)

# %%
import codecs
import collections

postag_map = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
postag_reverse_map = {item[1]:item[0] for item in postag_map.items()}

#pos_lang_list = ["eng-x-bible-mixed", "deu-x-bible-bolsinger", "rus-x-bible-newworld", "dan-x-bible-newworld", "fin-x-bible-helfi", 'gle-x-bible', 'urd-x-bible-2007',
#'pol-x-bible-newworld', "swe-x-bible-newworld"
#, "ita-x-bible-2009"
#, "fra-x-bible-louissegond", "spa-x-bible-newworld", "zho-x-bible-newworld", "arb-x-bible", "tam-x-bible-newworld"]

pos_lang_list = ["eng-x-bible-mixed", "spa-x-bible-newworld", "fra-x-bible-louissegond", "deu-x-bible-bolsinger", "rus-x-bible-newworld", "arb-x-bible"]

pos_val_lang_list = ['hun-x-bible-newworld', 'ell-x-bible-newworld', 'heb-x-bible-helfi', 'ces-x-bible-newworld'
# , 'ita-x-bibble-2009', "fin-x-bible-helfi"
]
#pos_test_lang_list = ['yor-x-bible-2010', 'por-x-bible-newworld1996', 'hin-x-bible-bsi', 'pes-x-bible-newmillennium2011', 'tur-x-bible-newworld', 'ind-x-bible-newworld' ]
pos_test_lang_list = [
'por-x-bible-newworld1996',
'hin-x-bible-bsi',
'pes-x-bible-newmillennium2011',
'tur-x-bible-newworld',
'ind-x-bible-newworld',
'yor-x-bible-2010',
'afr-x-bible-newworld',
'amh-x-bible-newworld',
'eus-x-bible-navarrolabourdin',
'bul-x-bible-newworld',
'lit-x-bible-ecumenical',
'tel-x-bible',
'bam-x-bible',
'bel-x-bible-bokun',
'myv-x-bible',
'glv-x-bible',
'mar-x-bible'
]

def get_db_nodecount(dataset):
	res = 0
	for lang in dataset.nodes_map.values():
		for verse in lang.values():
			res += len(verse)
	
	return res

def get_language_nodes(dataset, lang_list, sentences):
	node_count = get_db_nodecount(dataset)
	pos_labels = {}

	pos_node_cover = collections.defaultdict(list)
	for lang in lang_list:
		if lang in dataset.nodes_map:
			for sentence in sentences:
				if sentence not in pos_labels:
					pos_labels[sentence] = torch.zeros(dataset.verse_lengthes[sentence], len(postag_map))
				if sentence in dataset.nodes_map[lang]:
					for tok in dataset.nodes_map[lang][sentence]:
						pos_node_cover[sentence].append(dataset.nodes_map[lang][sentence][tok])
	
	return pos_labels, pos_node_cover

def create_structures(dataset, all_tags):
	pos_labels = {} 
	pos_node_cover = collections.defaultdict(list)

	for lang in all_tags:
		for sent_id in all_tags[lang]:
			sent_tags = all_tags[lang][sent_id]
			if sent_id not in pos_labels:
				pos_labels[sent_id] = torch.zeros(dataset.verse_lengthes[sent_id], len(postag_map))
			for w_i in range(len(sent_tags)):
				if w_i not in dataset.nodes_map[lang][sent_id]:
					continue
				pos_labels[sent_id][dataset.nodes_map[lang][sent_id][w_i], sent_tags[w_i]] = 1
				pos_node_cover[sent_id].append(dataset.nodes_map[lang][sent_id][w_i])
	return pos_labels, pos_node_cover

def get_pos_tags(dataset, pos_lang_list):
	all_tags = {}
	for lang in pos_lang_list:
		if lang not in dataset.nodes_map:
			continue
		all_tags[lang] = {}

		if os.path.exists(f'/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/{lang}.conllu'):
			base_path = '/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/'
		elif os.path.exists(f'/mounts/data/proj/ayyoob/POS_tagging/baselines/{lang}.conllu'):
			base_path = '/mounts/data/proj/ayyoob/POS_tagging/baselines/'
		else:
			print('Warning file not found:', lang)
			
		# if os.path.exists(F"/mounts/work/silvia/POS/TAGGED_LANGS/{lang}.conllu"):	
		# 	base_path = F"/mounts/work/silvia/POS/TAGGED_LANGS/"
		# else:
		# 	base_path = F"/mounts/work/mjalili/projects/gnn-align/data/pbc_pos_tags/"
		
		with codecs.open(F"{base_path}{lang}.conllu", "r", "utf-8") as lang_pos:
			tag_sent = []
			sent_id = ""
			for sline in lang_pos:
				sline = sline.strip()
				if sline == "":
					if sent_id not in dataset.nodes_map[lang]:
						tag_sent = []
						sent_id = ""
						continue

					all_tags[lang][sent_id] = [postag_map[p[3]] for p in tag_sent]
					tag_sent = []
					sent_id = ""
				elif "# verse_id" in sline or '# sent_id' in sline:
					sent_id = sline.split()[-1]
				elif sline[0] == "#":
					continue
				else:
					tag_sent.append(sline.split("\t"))

	pos_labels, pos_node_cover = create_structures(dataset, all_tags)
	return pos_labels, pos_node_cover
	

def get_pos_tags_from_bronze_data(dataset, file_path, language):
	file_content = torch.load(file_path)
	all_tags = {language:{}}

	node_count = get_db_nodecount(dataset)
	pos_labels = torch.zeros(node_count, len(postag_map))
	pos_node_cover = collections.defaultdict(list)

	for sent_id in file_content:
		if sent_id in dataset.nodes_map[language]:
			m_list = file_content[sent_id].items() if isinstance(file_content[sent_id], dict) else file_content[sent_id]
			for item in m_list:
				pos_labels[ dataset.nodes_map[language][sent_id][item[0]], item[1]] = 1
				pos_node_cover[sent_id].append(dataset.nodes_map[language][sent_id][item[0]])
	
	return pos_labels, pos_node_cover

def read_ud_gold_file(f_path, w2v_model, lang):
	pos_labels = torch.zeros(w2v_model.wv.vectors.shape[0], len(postag_map))
	with codecs.open(f_path, "r", "utf-8") as lang_pos:
			for sline in lang_pos:
				sline = sline.strip()
				if sline == "":
					pass
				elif "# verse_id" in sline:
					pass
				elif sline[0] == "#":
					continue
				else:
					line_items = list(sline.split("\t"))
					word = line_items[1]
					tag = line_items[3]
					try:
						# print(f'{lang}:{word.lower()}')
						idx = w2v_model.wv.key_to_index[f'{lang}:{word.lower()}']
						
					except Exception as e: # some words from the gold data may not exist in bible. we just skip them
						continue
					
					if tag == '_':
						#print('tag is', tag)
						continue

					pos_labels[idx, postag_map[tag]] += 1
	
	index = (torch.sum(pos_labels, dim=1) > 0.1).nonzero()
	
	maxes, tags = torch.max(pos_labels, dim=1)
	print(torch.sum(pos_labels))

	return tags[index], index


# %%
import torch
print(torch.__version__)

torch.hub.set_dir('/mounts/work/ayyoob/saved_models/torch/cache')
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()
print('done')


# %%

def get_xlmr_embeddings(nodes_map, verse, languages, w2v_model, nodes_count, xlmr, save_path, x):
    to_save = {}
    if os.path.exists(f'{save_path}/xlmr_features3/{verse}.torch.bin'):
        to_save = torch.load(f'{save_path}/xlmr_features3/{verse}.torch.bin')
        return to_save['res']

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
                try:
                    token = w2v_model.wv.index_to_key[x[tok_id].item()][4:] # TODO fixme eng:apple
                except:
                    print(verse)
                    print(x[tok_id].item())
                    print(w2v_model.wv.index_to_key[x[tok_id].item()])
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



#verse = train_dataset.accepted_verses[1008]
#verse_info = torch.load(utils.graph_dataset_path + path_ext + f'/verses/{verse}_info.torch.bin')
#get_xlmr_embeddings(train_dataset.nodes_map, verse, current_editions, w2v_model, verse_info['x'].size(0), xlmr,  utils.graph_dataset_path + path_ext + '/')
#train_dataset.features.append(afeatures.ForwardFeature( 512, 1024, 'xlmr'))

# %%
verse = '01001001'
if (os.path.exists(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
    and os.path.getsize(utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin") > 0):
    
    verse_info = torch.load( utils.graph_dataset_path + path_ext + f"/verses/{verse}_info.torch.bin")
    
    x = verse_info['x']
    
    if not torch.is_tensor(x) or x.size(1) < 10:
        print('going to do')
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
    print('notexisting')

# %%
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random, time 
from multiprocessing import Pool


class POSTAGGNNDataset(Dataset):

    def __init__(self, dataset, verses, edit_files, alignments, node_cover, pos_labels, data_dir, create_data=False, group_size = 20, transformer=True):
        self.node_cover = node_cover
        self.pos_labels = pos_labels
        self.data_dir = data_dir
        self.items = self.calculate_size(verses, group_size, node_cover)
        self.dataset = dataset
        self.editions = edit_files

        if create_data:
            self.calculate_verse_stats(verses, edit_files, alignments, dataset, data_dir)

        #self.pool = Pool(4)         
        self.transformer = transformer
    
    def set_transformer(self, transformer):
        self.transformer = transformer

    def calculate_size(self, verses, group_size, node_cover):
        res = []
        self.res_new_testament = []
        self.res_old_testament = []
        for verse in verses:
            covered_nodes = node_cover[verse]
            random.shuffle(covered_nodes)
            items = [covered_nodes[i:i + group_size] for i in range(0, len(covered_nodes), group_size)]
            res.extend([(verse, i) for i in items])
 
            if verse in new_testament_verses:
                self.res_new_testament.extend([(verse, i) for i in items])
            if verse in old_testament_verses:
                self.res_old_testament.extend([(verse, i) for i in items])


        return res
    
    def __len__(self):
        global testaments
        if testaments == 'new':
            return len(self.res_new_testament)
        elif testaments == 'old':
            return len(self.res_old_testament)
        return len(self.items)

    
    def __getitem__(self, idx):
        global testaments
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if testaments == 'new':
            verse, nodes = self.res_new_testament[idx]
        elif testaments == 'old':
            verse, nodes = self.res_old_testament[idx]
        else:
            verse, nodes = self.items[idx]
        
        self.verse_info = {verse: torch.load(f'{self.data_dir}/verses/{verse}_info.torch.bin')}


        padding = self.verse_info[verse]['padding']
        
        if self.transformer:
            language_based_nodes, transformer_indices = posutil.get_language_based_nodes(self.dataset.nodes_map, verse, nodes, padding)
        else:
            language_based_nodes, transformer_indices = 0,0
        
        if XLMR:
            xlmr_emb = get_xlmr_embeddings(self.dataset.nodes_map, verse, self.editions, w2v_model, self.verse_info[verse]['x'].size(0), xlmr, self.data_dir, self.verse_info[verse]['x'][:, 9].long())
            self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], xlmr_emb), dim=1)
        # # Add POSTAG to set of features
        if POSTAG:
            postags = self.pos_labels[verse][padding: self.verse_info[verse]['x'].size(0) + padding, : ]
            postags = postags.detach().clone()
            postags[torch.LongTensor(nodes) - padding, :] = 0
            self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], postags), dim=1)

        # Add token id as a feature, used to extract token information (like token's tag distribution)
        word_number = self.verse_info[verse]['x'][:, 9]
        word_number = torch.unsqueeze(word_number, 1)
        self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], word_number), dim=1)
        edge_index = self.verse_info[verse]['edge_index'] 
        #print('dataset time:', time.time()-start_time)

        if not WORDEMBEDDING:
            self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'][:, :9], self.verse_info[verse]['x'][:, 10:]), dim=1)
        return {'verse':verse, 'x':self.verse_info[verse]['x'], 'edge_index': edge_index if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long), 
                'pos_classes': self.pos_labels[verse][nodes, :], 'pos_index': torch.LongTensor(nodes) - padding, 
                'padding': padding, 'lang_based_nodes': language_based_nodes, 'transformer_indices': transformer_indices}

def create_me_a_gnn_dataset_you_stupid(node_covers, labels, group_size=100, editions=current_editions, verses=train_dataset.accepted_verses):

    train_ds = POSTAGGNNDataset(train_dataset, verses, editions, {}, node_covers[0], labels[0], utils.graph_dataset_path + path_ext + '/', group_size=group_size)

    return train_ds

def add_source_labels(test_labels, source_labels):
    for verse in test_labels:
        if verse in source_labels:
            test_labels[verse] += source_labels[verse]

train_pos_labels, train_pos_node_cover = get_pos_tags(train_dataset, pos_lang_list)
val_pos_labels, val_pos_node_cover = get_pos_tags(train_dataset, pos_val_lang_list)
add_source_labels(val_pos_labels, train_pos_labels)


#test_gnn_datasets = {}
#test_data_loaders = {}
#for test_lang in pos_test_lang_list:
#    if 'yor' not in test_lang:
#        test_pos_labels, test_node_cover = get_pos_tags(train_dataset, [test_lang])
#        test_gnn_datasets[test_lang] = create_me_a_gnn_dataset_you_stupid([test_node_cover], [test_pos_labels], group_size=10000, verses=shuffled_verses[:500])
#        test_data_loaders[test_lang] = DataLoader(test_gnn_datasets[test_lang], batch_size=1, shuffle=False)


gnn_dataset_train_pos= create_me_a_gnn_dataset_you_stupid([train_pos_node_cover], [train_pos_labels], group_size=8, verses= train_dataset.accepted_verses[:])

gnn_dataset_train_pos_bigbatch = create_me_a_gnn_dataset_you_stupid([train_pos_node_cover], [train_pos_labels], group_size=8, verses= shuffled_verses[:])
gnn_dataset_val_pos = create_me_a_gnn_dataset_you_stupid([val_pos_node_cover], [val_pos_labels], group_size=10000, verses=shuffled_verses[:1000])
gnn_dataset_test_pos = create_me_a_gnn_dataset_you_stupid([val_pos_node_cover], [val_pos_labels], group_size=10000, verses=train_dataset.accepted_verses)

train_data_loader_bigbatch = DataLoader(gnn_dataset_train_pos_bigbatch, batch_size=1, shuffle=False)
val_data_loader_pos = DataLoader(gnn_dataset_val_pos, batch_size=1, shuffle=False)
test_data_loader_pos = DataLoader(gnn_dataset_test_pos, batch_size=1, shuffle=False)


# %%
def get_data_loadrs_for_target_editions(target_editions, dataset, verses, data_dir, transformer):
    target_pos_labels, target_pos_node_cover = get_language_nodes(dataset, target_editions, verses)
    gnn_dataset_target_pos = POSTAGGNNDataset(dataset, verses, None, {}, target_pos_node_cover, target_pos_labels, data_dir, group_size = 50000, transformer=transformer)
    target_data_loader = DataLoader(gnn_dataset_target_pos, batch_size=1, shuffle=False)
    
    return target_data_loader

# %%
import pprint
#train_dataset.features[-1].out_dim = 100
for i,feat in enumerate(train_dataset.features[:]):
    #feat.out_dim = feat.out_dim*4
    print(i, vars(feat))

# %%
def save_model(model, name):
    model.encoder.feature_encoder.feature_types[0] = afeatures.OneHotFeature(20, 35, 'editf')
    model.encoder.feature_encoder.feature_types[1] = afeatures.OneHotFeature(32, 256, 'position')
    model.encoder.feature_encoder.feature_types[2] = afeatures.FloatFeature(4, 'degree_centrality')
    model.encoder.feature_encoder.feature_types[3] = afeatures.FloatFeature(4, 'closeness_centrality')
    model.encoder.feature_encoder.feature_types[4] = afeatures.FloatFeature(4, 'betweenness_centrality')
    model.encoder.feature_encoder.feature_types[5] = afeatures.FloatFeature(4, 'load_centrality')
    model.encoder.feature_encoder.feature_types[6] = afeatures.FloatFeature(4, 'harmonic_centrality')
    model.encoder.feature_encoder.feature_types[7] = afeatures.OneHotFeature(32, 256, 'greedy_modularity_community')
    model.encoder.feature_encoder.feature_types[8] = afeatures.OneHotFeature(32, 256, 'community_2')
    #model.encoder.feature_encoder.feature_types[9] = afeatures.MappingFeature(100, 'word')
    #model.encoder.feature_encoder.feature_types[10] = afeatures.MappingFeature(len(postag_map), 'tag_priors', freeze=True)
    torch.save(model, f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/POSTAG_{name}_' + '.pickle')

# %%
def test_existing_model(model_path, use_transformer):
    global model
    model = torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/{model_path}', map_location=torch.device('cpu'))
    model.to(dev)
    print(model_path)
    if use_transformer:
        model.decoder.to(dev2)

    test_mostfreq(yor_data_loader_heb, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
    #test(1, yorbronz_data_loader, True)
    #test(1, engtest_data_loader, True)

# %%
yor_gold_mostfreq_tag, yor_gold_mostfreq_index = read_ud_gold_file('/mounts/work/silvia/POS/yo_ytb-ud-test.conllu', w2v_model, 'yor')
yor_data_loader = get_data_loadrs_for_target_editions(['yor-x-bible-2010'], train_dataset, shuffled_verses[:2000], utils.graph_dataset_path+ path_ext + '/', transformer=True)


tam_gold_mostfreq_tag, tam_gold_mostfreq_index = read_ud_gold_file('/mounts/work/silvia/POS/ta_mwtt-ud-test.conllu', w2v_model, 'tam')
tam_data_loader = get_data_loadrs_for_target_editions(['tam-x-bible-newworld'], train_dataset, shuffled_verses[:2000], utils.graph_dataset_path+ path_ext + '/', transformer=True)

arb_gold_mostfreq_tag, arb_gold_mostfreq_index = read_ud_gold_file('/nfs/datx/UD/ar_pud-ud-test.conllu', w2v_model, 'arb')
arb_data_loader = get_data_loadrs_for_target_editions(['arb-x-bible'], train_dataset, shuffled_verses[:2000], utils.graph_dataset_path+ path_ext + '/', transformer=True)

por_gold_mostfreq_tag, por_gold_mostfreq_index = read_ud_gold_file('/nfs/datx/UD/pt_pud-ud-test.conllu', w2v_model, 'por')
por_data_loader = get_data_loadrs_for_target_editions(['por-x-bible-newworld1996'], train_dataset, shuffled_verses[:2000], utils.graph_dataset_path+ path_ext + '/', transformer=True)

#test_existing_model('pos_tagging_posfeatFalse_transformerFalse_trainWEFalse_maskLangTrue_20220209-190017.pickle', False)

#test_mostfreq(yorbronz_data_loader, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)), from_target=True)
#test_mostfreq(tam_data_loader_grc, True, tam_gold_mostfreq_tag, tam_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))


# %%
importlib.reload(posutil)
threshold = 0.8
type_check = True
from copy import deepcopy
tag_based_threshold = 1
testaments = all

#target_editions = []

#for edition in current_editions:
#    if edition not in pos_lang_list: # and (edition in pos_val_lang_list or edition in pos_test_lang_list):
#        target_editions.append(edition)

#for verse in train_dataset.accepted_verses:
#    if verse not in train_pos_labels:
#        train_pos_labels[verse] = torch.zeros(train_dataset.verse_lengthes[verse], len(postag_map))

#m_path = 'POSTAG_15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.85_TypchckTrue_TgBsdSlctTrue_tstamtall_20220428-121249_ElyStpDlta0-GA-chnls1024_small_.pickle'
#model = torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/{m_path}', map_location=torch.device('cpu'))
#model.to(dev)
## model.decoder.to(dev2)
#XLMR = True
#POSTAG = True
#WORDEMBEDDING = False
#criterion = nn.CrossEntropyLoss()
#gnn_dataset_val_pos.set_transformer(False)
#test(0, val_data_loader_pos, True) 
## model.model_name = '/mounts/work/ayyoob/models/gnn/checkpoint/postagging/pos_tagging_15langs-nopersian_posfeatFalsealltargets_transformerFalse6layresresidualFalse_trainWEFalse_maskLangTrue_epoch1__20220318-092426_earlystoppingdelta0-GA-channels512_.pickle'


#gnn_dataset_train_pos_bigbatch.set_transformer(False)
#target_data_loader_train = get_data_loadrs_for_target_editions(target_editions, train_dataset, train_dataset.accepted_verses, utils.graph_dataset_path + path_ext + '/', transformer=False)

#res_ = torch.load(f'/mounts/work/ayyoob/results/gnn_postag/data/15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResFalse_trainWEFalse_mskLngTrue_E1_traintgt0_TypchckTrue_TgBsdSlctTrue1_20220331-231532_ElyStpDlta0-GA-chnls512_small_th0.9_typchkFalse.torch.bin')
#tag_frequencies, tag_frequencies_target, pos_node_cover_exts, pos_label_exts, tag_based_stats = res_
#tag_frequencies_source = tag_frequencies - tag_frequencies_target
##train_pos_node_covers_ext, train_pos_labels_ext = pos_node_cover_exts[0], pos_label_exts[0]
#res_ = posutil.get_tag_frequencies_node_tags(model, [train_pos_node_cover], [train_pos_labels], len(postag_map), w2v_model.wv.vectors.shape[0],
#                                [target_data_loader_train], [train_data_loader_bigbatch], DataEncoder,
#                                target_train_treshold=threshold, type_check=type_check, source_tag_frequencies=tag_frequencies_source)

#torch.save(res_, f'/mounts/work/ayyoob/results/gnn_postag/data/{model.model_name}_th{threshold}_typchk{type_check}.torch.bin')
##res_ = torch.load(f'/mounts/work/ayyoob/results/gnn_postag/data/15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResFalse_trainWEFalse_mskLngTrue_E1_traintgt0_TypchckTrue_TgBsdSlctTrue1_20220331-231532_ElyStpDlta0-GA-chnls512_small_th0.9_typchkFalse.torch.bin')
#tag_frequencies, tag_frequencies_target, pos_node_cover_exts, pos_label_exts, tag_based_stats = res_
#train_pos_node_covers_ext, train_pos_labels_ext = pos_node_cover_exts[0], pos_label_exts[0]
###tag_frequencies_source = tag_frequencies - tag_frequencies_target


##def add_target_language_training_nodes(tag_based_stats, train_pos_nod_cover, train_pos_labels, tag_frequencies_source, top_tags_threshold=0.05):
##  node_cover_copy = deepcopy(train_pos_nod_cover)
##  pos_label_copy = deepcopy(train_pos_labels)
##  source_tag_frequency_sum = torch.sum(tag_frequencies_source)
##  source_tag_based_probs = torch.sum(tag_frequencies_source, dim=0) / source_tag_frequency_sum

##  print('source tag based probs:', source_tag_based_probs)

##  total_node_count = 0
##  for tag, tag_list in tag_based_stats.items():
##      total_node_count += len(tag_list)
##      tag_based_stats[tag] = sorted(tag_list, key=lambda x: x[2], reverse=True)
    
##  for tag, tag_list in tag_based_stats.items():
##      tags_target_count = source_tag_based_probs[tag] * total_node_count * top_tags_threshold
##      tags_half_location = len(tag_list) * (top_tags_threshold)
##      print(tag, postag_reverse_map[tag], tags_target_count, tags_half_location)

##      for item in tag_list[:int(min(tags_target_count, tags_half_location))]:
##          node_cover_copy[item[0]].append(item[1])
##          pos_label_copy[item[0]][item[1], tag] = 1
    
##  for verse in node_cover_copy:
##      if random.randint(0, 1000) == 1:
##          print(len(train_pos_node_cover[verse]), len(node_cover_copy[verse]), torch.sum(train_pos_labels[verse]), torch.sum(pos_label_copy[verse]))
    
##  return node_cover_copy, pos_label_copy

###train_pos_node_covers_ext, train_pos_labels_ext = add_target_language_training_nodes(tag_based_stats, train_pos_node_cover, train_pos_labels, tag_frequencies_source, top_tags_threshold=tag_based_threshold)

###torch.save((tag_frequencies, tag_frequencies_target, train_pos_node_covers_ext, train_pos_labels_ext), 
###f'/mounts/work/ayyoob/results/gnn_postag/data/pos_tagging_15langs-nopersian_posfeatFalsealltargets_transformerFalse6layresresidualFalse_trainWEFalse_maskLangTrue_epoch1__20220318-092426_earlystoppingdelta0-GA-channels512_th0.95_typecheckFalse_extendedtags0.1and0.1')
### tag_frequencies, tag_frequencies_target, train_pos_node_covers_ext, train_pos_labels_ext = 
### torch.load('/mounts/work/ayyoob/results/gnn_postag/data/15langs-nopersian_posfeatTruealltargets_transformerFalse6layresresidualFalse_trainWEFalse_maskLangTrue_epoch1_traintarget0.95_typecheckFalse_20220319-111011_earlystoppingdelta0-GA-channels512_th0.95_typecheckFalse_extendedtagsForValTest0.1and0.1')

#tag_frequencies_source = tag_frequencies - tag_frequencies_target
# # print(1, torch.sum(tag_frequencies_target))
# # posutil.keep_only_type_tags(tag_frequencies_target)
# # print(1, torch.sum(tag_frequencies_target))
#word_frequencies_target = torch.sum(tag_frequencies_target.to(torch.device('cpu')), dim=1)
#tag_frequencies = tag_frequencies_source + tag_frequencies_target
#tag_frequencies_copy = tag_frequencies.detach().clone()

#tag_frequencies_copy[torch.logical_and(word_frequencies_target>0.1, word_frequencies_target<3), :] = 0.0000001

## We have to give uniform noise to some training examples to prevent the model from returning one of the most frequent tags always!!
#uniform_noise = torch.BoolTensor(tag_frequencies.size(0))
#uniform_noise[:] = True
#shuffle_tensor = torch.randperm(tag_frequencies.size(0))[:int(tag_frequencies.size(0)*0.7)]
#uniform_noise[shuffle_tensor] = False
#tag_frequencies_copy[torch.logical_and(uniform_noise, word_frequencies_target < 0.1), :] = 0.0000001

#sm = torch.sum(tag_frequencies_copy, dim=1)
#normalized_tag_frequencies = (tag_frequencies_copy.transpose(1,0) / sm).transpose(1,0)


# %%
importlib.reload(posutil)
threshold = 0.85
type_check = False
testaments = all



distributions = torch.load(utils.graph_dataset_path + path_ext + '/distributions_all.torch.bin')
print(len(distributions), distributions.keys())

normalized_tag_frequencies, tag_frequencies = posutil.get_sourceside_words_tag_frequence(distributions, w2v_model, w2v_model.wv.vectors.shape[0], len(postag_map))



# %%
importlib.reload(posutil)
threshold = 0.95
type_check = True
target_editions = []

for edition in current_editions:
   if edition not in pos_lang_list: # and (edition in pos_val_lang_list or edition in pos_test_lang_list):
       target_editions.append(edition)

for verse in train_dataset.accepted_verses:
   if verse not in train_pos_labels:
       train_pos_labels[verse] = torch.zeros(train_dataset.verse_lengthes[verse], len(postag_map))

gnn_dataset_train_pos_bigbatch.set_transformer(False)
target_data_loader_train = get_data_loadrs_for_target_editions(target_editions, train_dataset, train_dataset.accepted_verses, utils.graph_dataset_path + path_ext + '/', transformer=False)


m_path = 'POSTAG_6lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.85_TypchckFalse_TgBsdSlctTrue_tstamtall_ramylangs_20220512-112432_ElyStpDlta0-GA-chnls1024_small_final_.pickle'
model = torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/{m_path}', map_location=torch.device('cpu'))
model.to(dev)
# model.decoder.to(dev2)
model_name = model.model_name
XLMR = True
POSTAG = True
WORDEMBEDDING = False
mask_language = True
criterion = nn.CrossEntropyLoss()
gnn_dataset_val_pos.set_transformer(False)
test(0, val_data_loader_pos, True) 

#train_pos_node_covers_ext, train_pos_labels_ext = posutil.get_self_learning_data(tag_frequencies, model, w2v_model.wv.vectors.shape[0], len(postag_map), target_data_loader_train, DataEncoder, threshold, train_pos_node_cover, train_pos_labels)
#torch.save((normalized_tag_frequencies, tag_frequencies, train_pos_node_covers_ext, train_pos_labels_ext), f'/mounts/work/ayyoob/results/gnn_postag/data/{model.model_name}')
#normalized_tag_frequencies, tag_frequencies, train_pos_node_covers_ext, train_pos_labels_ext = torch.load(f'/mounts/work/ayyoob/results/gnn_postag/data/{model.model_name}')

gnn_dataset_test_pos.set_transformer(False)
gen_langs = pos_test_lang_list[:]
gen_langs.extend(pos_val_lang_list)
for lang in gen_langs:
    posutil.generate_target_lang_tags_onedataset(model, lang, model_name, mask_language,
            train_dataset, test_data_loader_pos, DataEncoder, postag_map['X'])


# %%
from tqdm import tqdm
from torchvision import models

def create_model(train_gnn_dataset, test_gnn_dataset,
                tag_frequencies=False, use_transformers=False, train_word_embedding=False, mask_language=True, residual_connection = False,
                 params=''):
    global model, criterion, optimizer, early_stopping, start_time

    if WORDEMBEDDING:
        features = train_dataset.features[:]
        features[-1].out_dim=100
    else:
        features = train_dataset.features[:-1]
    
    #features.append(afeatures.PassFeature(name='posTAG', dim=len(postag_map)))
    if XLMR:
        features.append(afeatures.PassFeature(1024, 'xlmr'))
    if POSTAG:
        features.append(afeatures.PassFeature(len(postag_map), 'neighbor_tags'))
    if tag_frequencies:
        features.append(afeatures.MappingFeature(len(postag_map), 'tag_priors', freeze=True))
        #features[-2] = afeatures.MappingFeature(len(postag_map), 'tag_priors', freeze=True)
    #features[9].freeze = not train_word_embedding
    for i,feature in enumerate(features):
        print(i, vars(feature))

    train_gnn_dataset.set_transformer(use_transformers)
    test_gnn_dataset.set_transformer(use_transformers)
    train_data_loader = DataLoader(train_gnn_dataset, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_gnn_dataset, batch_size=1, shuffle=False)

    clean_memory()
    drop_out = 0
    n_head = 1
    in_dim = sum(t.out_dim for t in features)
    print(in_dim)
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    delta = 0
    patience = 8
    early_stopping = EarlyStopping(verbose=True, path=f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/check_point_{start_time}.pt', patience=patience, delta=delta)
    channels = 1024
    decoder_in_dim = n_head * channels + in_dim if residual_connection else 0

    print('len features', len(features), f'start time: {start_time}')

    if use_transformers:
        decoder = POSDecoderTransformer(decoder_in_dim, 2048, len(postag_map), residual_connection, features[1].n_classes, drop_out=drop_out).to(dev2)
    else:
        decoder = POSDecoder(decoder_in_dim, decoder_in_dim*2, len(postag_map), residual_connection)
        
    model = pyg_nn.GAE(Encoder(in_dim, channels, features, n_head, has_tagfreq_feature=tag_frequencies), decoder).to(dev)


    if use_transformers:
        decoder.to(dev2)

    criterion = nn.CrossEntropyLoss(ignore_index=postag_map['X'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    torch.set_printoptions(edgeitems=10)
    print("model params - decoder params - conv1", sum(p.numel() for p in model.parameters()), sum(p.numel() for p in decoder.parameters()))

    for epoch in range(1, 2):
        print(f"\n----------------epoch {epoch} ---------------")
        
        train(epoch, train_data_loader, mask_language, test_data_loader)

        if early_stopping.early_stop:
            model.load_state_dict(torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/check_point_{start_time}.pt'))

        model_name = f'{len(pos_lang_list)}lngs-POSFeat{tag_frequencies}alltgts_trnsfrmr{use_transformers}6LRes{residual_connection}_trainWE{train_word_embedding}_mskLng{mask_language}_E{epoch}_{params}_{start_time}_ElyStpDlta{delta}-GA-chnls{channels}_{path_ext[1:]}'
        model.model_name = model_name
        save_model(model, model_name)
        test(epoch, test_data_loader, mask_language) 
        # test_mostfreq(yor_data_loader, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
        # test_mostfreq(tam_data_loader, True, tam_gold_mostfreq_tag, tam_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
        # test_mostfreq(arb_data_loader, True, arb_gold_mostfreq_tag, arb_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
        # test_mostfreq(por_data_loader, True, por_gold_mostfreq_tag, por_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))

        clean_memory()
    
    return model_name


# %%
def filter_node_cover_for_languages(nodes_cover, nodes_map, languages):
    res = {}
    for verse in tqdm(nodes_cover):
        res[verse] = []
        for node in nodes_cover[verse]:
            for lang in languages:
                if verse in nodes_map[lang] and node in nodes_map[lang][verse].values():
                    res[verse].append(node)
                    break
    return res


# %%
# Final Bronze3, with xlmr, ramy training langs
importlib.reload(posutil)
XLMR = False
POSTAG = True
WORDEMBEDDING = True
testaments = 'all'
mask_language = True
start_time = "20220512-122144"

#lang = pos_test_lang_list[int(sys.argv[1])]
#for lang in pos_test_lang_list:
train_langs = pos_val_lang_list[:]
train_langs.extend(pos_lang_list)
train_langs.append(lang)
params = f'traintgt{threshold}_Typchck{type_check}_TgBsdSlct{True}_RAMY'
#filtered_nodes_cover = filter_node_cover_for_languages(train_pos_node_covers_ext, train_dataset.nodes_map, train_langs)

gnn_dataset_train_pos_ext = create_me_a_gnn_dataset_you_stupid([train_pos_node_covers_ext], [train_pos_labels_ext], group_size=8)
model_name = create_model(gnn_dataset_train_pos_ext, gnn_dataset_val_pos,
        train_word_embedding=False, mask_language=mask_language, use_transformers=True,
        tag_frequencies=False, params=params, residual_connection=True)


gen_langs = pos_test_lang_list[:]
gen_langs.extend[pos_val_lang_list]
for lang in gen_langs:
    posutil.generate_target_lang_tags_onedataset(model, lang, model_name, mask_language,
            train_dataset, test_data_loader_pos, DataEncoder, postag_map['X'])

model = None
decoder = None
clean_memory()
    
1/0

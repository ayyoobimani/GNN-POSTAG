from turtle import pos
import collections
from numpy import source
import torch
from copy import deepcopy
from tqdm import tqdm


def create_indices(items, train_nodes, padding, lang_ind):
    items = sorted(items, key=lambda i: i[0])
    
    to_add = []
    reverse_indices = []
    for i, it in enumerate(items):
        to_add.append(it[1] - padding)
        if it[1] in train_nodes:
            index = train_nodes.index(it[1])
            reverse_indices.append((index, i, lang_ind))

    return to_add, reverse_indices
def get_language_based_nodes2(nodes_map, verse, train_nodes, padding, pool, langs=None):
    res = []
    transformer_indices = [[-1 for i in range(len(train_nodes))],[-1 for i in range(len(train_nodes))]]
    args = []

    if langs == None:
        langs = list(nodes_map.keys())

    lang_ind = 0
    for lang in langs:
        if verse in nodes_map[lang]:
            items = list(nodes_map[lang][verse].items())
            args.append((items, train_nodes[:], padding, lang_ind))
            lang_ind += 1
    
    res_ = pool.starmap(create_indices, args)
    for item in res_:
        to_add = item[0]
        for ri in item[1]:
            index = ri[0]
            i = ri[1]
            lang_ind = ri[2]
            transformer_indices[0][index] = lang_ind
            transformer_indices[1][index] = i
        res.append(to_add)
            
    return res, transformer_indices


def get_language_based_nodes(nodes_map, verse, train_nodes, padding, langs=None):
    res = []
    transformer_indices = [[-1 for i in range(len(train_nodes))],[-1 for i in range(len(train_nodes))]]
    
    train_nodes_map = { item:i for i,item in enumerate(train_nodes)}
    if langs == None:
        langs = list(nodes_map.keys())

    lang_ind = 0
    for lang in langs:
        if verse in nodes_map[lang]:
            items = nodes_map[lang][verse].items()
            sorted_items = [0 for i in range(len(items))]
            for item in items:
                sorted_items[item[0]] = item
            items = sorted_items

            to_add = []
            for i, it in enumerate(items):
                if it != 0:
                    to_add.append(it[1] - padding)

                    if it[1] in train_nodes_map:
                        index = train_nodes_map[it[1]]
                        transformer_indices[0][index] = lang_ind
                        transformer_indices[1][index] = i

            res.append(to_add)
            lang_ind += 1
            
    return res, transformer_indices

def get_target_lang_postags(model, dataset, data_loader, edit, mask_language, encoder_class, X_tag_pos):
    model.eval()
    res = {}
    res2 = {}
    data_endoer = encoder_class(data_loader, model, mask_language)
    
    if edit in dataset.nodes_map:
        with torch.no_grad():

            for z, verse, _, batch in data_endoer:
                if verse in dataset.nodes_map[edit]:
                    x = batch['x'][0]
                    edge_index = batch['edge_index'][0]
                    index = []
                    toks = list(dataset.nodes_map[edit][verse].keys())
                    for i in toks:
                        index.append(dataset.nodes_map[edit][verse][i])

                    # print(batch['x'][0][:, 10].shape)

                    batch['lang_based_nodes'], batch['transformer_indices'] = get_language_based_nodes(dataset.nodes_map, verse, index, batch['padding'][0], langs=[edit])
                    index = torch.LongTensor(index) - batch['padding'][0]

                    # print(torch.max(model.encoder.feature_encoder.layers[10].emb._parameters['weight'][batch['x'][0][index, :][:, 10].long(), :], dim=1))

                    preds = model.decoder(z, index, batch)
                    preds[:, X_tag_pos]  = -100000

                    #if hasattr(model.decoder, 'transformer'):
                    #    #language_based_nodes = batch['lang_based_nodes'] # These two lines are for the transformer decoder, probably for normal decoder I have to delete the previouse lines
                    #    #preds = preds[language_based_nodes[0]]
                    #    preds = preds[index]

                    probabilities, predicted = torch.max(torch.softmax(preds, dim=1), 1)

                    res[verse] = [(toks[i], predicted[i].item(), probabilities[i].item(), torch.sum(edge_index[0,:] == index[i]).item()) for i in range(len(toks))]

    return res

def generate_target_lang_tags(model, target_lang, params, mask_language, train_dataset, grc_dataset, heb_dataset, blinker_dataset, train_dataloader, grc_dataloader, heb_dataloader, blinker_dataloader, encoder_class, X_tag_pos):
    target_pos_tags = {}
     
    res_ = get_target_lang_postags(model, train_dataset, train_dataloader, target_lang, mask_language, encoder_class, X_tag_pos)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, grc_dataset, grc_dataloader, target_lang, mask_language, encoder_class, X_tag_pos)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, heb_dataset, heb_dataloader, target_lang, mask_language, encoder_class, X_tag_pos)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, blinker_dataset, blinker_dataloader, target_lang, mask_language, encoder_class, X_tag_pos)
    target_pos_tags.update(res_)

    torch.save(target_pos_tags, f'/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_{target_lang}_{params}_maskLang{mask_language}.pickle')
    return target_pos_tags

def generate_target_lang_tags_onedataset(model, target_lang, params, mask_language, train_dataset, train_dataloader, encoder_class, X_tag_pos):
    print(f'generating data for ', target_lang, ', at ',  f'/mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_{target_lang}_{params}.pickle')
     
    target_pos_tags = get_target_lang_postags(model, train_dataset, train_dataloader, target_lang, mask_language, encoder_class, X_tag_pos)

    torch.save(target_pos_tags, f'/mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_{target_lang}_{params}.pickle')
    return target_pos_tags

# These are functions to calculate the frequency vectors and also update training data with predictions for target language tags

# This funcion update existing structures that are used for training with new predictions
def update_trainset_with_predictions(node_cover, pos_labels, padding, index, max_values, pos_tags, word_added_to_train_freq, word_pos, threshold, type_constraints):
    pos_tags = pos_tags.to('cpu')
    index = index.to('cpu')
    pos_labels = pos_labels.to('cpu')
    word_pos = word_pos.to('cpu')
    max_values = max_values.to('cpu')

    ## I should filter high repetition words here!
    #word_added_to_train_freq[word_pos.long()] += 1
    #accepted_by_frequency = word_added_to_train_freq[word_pos.long()] < 10
    #index = index[accepted_by_frequency]
    #pos_tags = pos_tags[accepted_by_frequency]
    #word_pos = word_pos[accepted_by_frequency]

    # I should filter high repetition words here!
    #word_added_to_train_freq[word_pos.long()] += 1
    #accepted_by_frequency = word_added_to_train_freq[word_pos.long()] < 10
    #index = index[accepted_by_frequency]
    #pos_tags = pos_tags[accepted_by_frequency]
    #word_pos = word_pos[accepted_by_frequency]
    if type_constraints != None:
        accepted_by_type_constraint = type_constraints[word_pos.long(), pos_tags] * max_values > threshold
        
        index = index[accepted_by_type_constraint]
        pos_tags = pos_tags[accepted_by_type_constraint]
    else:
        accepted_values = max_values > threshold
        index = index[accepted_values]
        pos_tags = pos_tags[accepted_values]
        word_pos = word_pos[accepted_values]

    index_global = index + padding
    node_cover.extend(index_global.tolist())
    pos_labels[index_global, pos_tags] = 1

def update_tag_based_stats(tag_based_stats, index, pos_tags, max_values, class_count, verse):
    for id,pos, val in zip(index, pos_tags, max_values):
        if pos.item() != class_count-1:
            tag_based_stats[pos.item()].append((verse, id, val.item()))
# This function iterates a dataset to creates the tag vectors and update training structures
# for source languages call once
# for target twice. First to calculate tag frequencies for types. second time providing the calculated max_tag for each type
def get_words_tag_frequence(model, word_count, class_count, data_loader, encoder_class, tag_frequencies=None, from_silver_data=False, node_cover=None,
     pos_labels=None, mask_language=True, target_train_treshold=0.9, type_constraints=None, tag_based_stats=None):
    
    res = tag_frequencies
    if res == None:
        res = torch.ones(word_count, class_count)
        res[:, :] = 0.0000001
    
    data_encoder = encoder_class(data_loader, model, mask_language)
    word_added_to_train_freq = torch.zeros(word_count)

    with torch.no_grad():

        for z, verse, i, batch in data_encoder:
            index = batch['pos_index'][0]
            
            if from_silver_data:
                tags_onehot = batch['pos_classes'][0]
            else:
                tags_onehot = model.decoder(z, index, batch)
            #print(tags_onehot)
            #print(torch.softmax(tags_onehot, dim=1))
            #break
            max_values, pos_tags = torch.max(torch.softmax(tags_onehot, dim=1), 1)
            word_pos = batch['x'][0][index, 9]

            if not from_silver_data:
                if node_cover != None:
                    update_trainset_with_predictions(node_cover[verse], pos_labels[verse], batch['padding'][0], index, max_values, pos_tags, word_added_to_train_freq, word_pos, target_train_treshold, type_constraints)
                if tag_based_stats != None:
                    update_tag_based_stats(tag_based_stats, index, pos_tags, max_values, class_count, verse)
                accepted_values = max_values > 0.0
                word_pos = word_pos[accepted_values]
                pos_tags = pos_tags[accepted_values]
            
            res[word_pos.long(), pos_tags.long()] += 1



    sm = torch.sum(res, dim=1)
    res_normalized = (res.transpose(1,0) / sm).transpose(1,0)
    
    return res_normalized, res


# Calls the above functions over any of the datasets (train, grc, heb, blinker) to get pos_tag vectors for each word. (once for source languages and once for target languages)
def get_tag_frequencies_node_tags(model, node_covers, pos_labels, class_count, word_count,
                                    target_data_loaders,
                                    data_loaders_bigbatch, encoder_class, target_train_treshold, source_tag_frequencies=None,
                                    tag_frequencies_target=None, type_check=False):
    node_cover_copies = []
    pos_label_copies = []
    tag_based_stats = collections.defaultdict(list)
    for nc, pl in zip(node_covers, pos_labels):
        node_cover_copies.append(deepcopy(nc))
        pos_label_copies.append(deepcopy(pl))

    if source_tag_frequencies == None:
        _, tag_frequencies = get_words_tag_frequence(model, word_count, class_count, data_loaders_bigbatch[0], encoder_class, from_silver_data=True)
        if len(data_loaders_bigbatch) > 1:
            for dlbb in data_loaders_bigbatch[1:]:
                get_words_tag_frequence(model, word_count, class_count, dlbb, encoder_class, from_silver_data=True, tag_frequencies=tag_frequencies)
    else:
        tag_frequencies = source_tag_frequencies
    
    #_, tag_frequencies = get_words_tag_frequence(model, word_count, class_count, blinker_data_loader_bigbatch, encoder_class, from_silver_data=True)

    if tag_frequencies_target == None:
        _, tag_frequencies_target = get_words_tag_frequence(model, word_count, class_count, target_data_loaders[0], encoder_class, node_cover=None if type_check else node_cover_copies[0], pos_labels=pos_label_copies[0], target_train_treshold=target_train_treshold, tag_based_stats=None)
        if len(target_data_loaders)>1:
            for tdl, ncc, plc in zip(target_data_loaders[1:], node_cover_copies[1:], pos_label_copies[1:]):
                get_words_tag_frequence(model, word_count, class_count, tdl, encoder_class, node_cover=None if type_check else ncc, pos_labels=plc, tag_frequencies=tag_frequencies_target, target_train_treshold=target_train_treshold, tag_based_stats=None)

        #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, encoder_class, node_cover=None if type_check else train_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy, target_train_treshold=target_train_treshold)


    if type_check:
        print('going to pick some training nodes based on type constraints')
        #probabilities = tag_frequencies_target/torch.sum(tag_frequencies_target, dim=1).view(-1,1)
        probabilities = tag_frequencies_target/torch.max(tag_frequencies_target, dim=1)[0].view(-1,1)
        sums = torch.sum(tag_frequencies_target, dim=1)
        probabilities[sums<12] = 0

        _, t = get_words_tag_frequence(model, word_count, class_count, target_data_loaders[0], encoder_class, node_cover=node_cover_copies[0], pos_labels=pos_label_copies[0], target_train_treshold=target_train_treshold, type_constraints=probabilities)
        if len(target_data_loaders)>1:
            for tdl, ncc, plc in zip(target_data_loaders[1:], node_cover_copies[1:], pos_label_copies[1:]):
                get_words_tag_frequence(model, word_count, class_count, tdl, encoder_class, node_cover=ncc, pos_labels=plc, target_train_treshold=target_train_treshold, type_constraints=probabilities, tag_frequencies=t)
    
    tag_frequencies += tag_frequencies_target.to(torch.device('cpu'))
        
    return tag_frequencies, tag_frequencies_target, node_cover_copies, pos_label_copies, tag_based_stats


def get_sourceside_words_tag_frequence(distributions, word2vec, word_count, class_count):
    res = torch.ones(word_count, class_count)
    res[:, :] = 0.0000001
    
    
    for l in distributions:
        for w in tqdm(distributions[l]):
            word = f'{l}:{w}'
            w_id = word2vec.wv.key_to_index[word]
            res[w_id] = distributions[l][w]

    sm = torch.sum(res, dim=1)
    res_normalized = (res.transpose(1,0) / sm).transpose(1,0)
    return res_normalized, res

def get_self_learning_data(tag_frequencies_target, model, word_count, class_count, target_data_loaders, encoder_class, target_train_treshold, node_covers, pos_labels):
    node_cover_copies = deepcopy(node_covers)
    pos_label_copies = deepcopy(pos_labels)

    probabilities = tag_frequencies_target/torch.max(tag_frequencies_target, dim=1)[0].view(-1,1)
    sums = torch.sum(tag_frequencies_target, dim=1)
    probabilities[sums<12] = 0

    get_words_tag_frequence(model, word_count, class_count, target_data_loaders, encoder_class, node_cover=node_cover_copies, pos_labels=pos_label_copies, target_train_treshold=target_train_treshold, type_constraints=probabilities)
    
    return node_cover_copies, pos_label_copies
from numpy import source
import torch

def get_language_based_nodes(nodes_map, verse, train_nodes, padding, langs=None):
    res = []
    transformer_indices = [[-1 for i in range(len(train_nodes))],[-1 for i in range(len(train_nodes))]]
    
    if langs == None:
        langs = list(nodes_map.keys())

    lang_ind = 0
    for lang in langs:
        if verse in nodes_map[lang]:
            items = nodes_map[lang][verse].items()
            items = sorted(items, key=lambda i: i[0])

            to_add = []
            for i, it in enumerate(items):
                to_add.append(it[1] - padding)
                if it[1] in train_nodes:
                    index = train_nodes.index(it[1])
                    transformer_indices[0][index] = lang_ind
                    transformer_indices[1][index] = i

            res.append(to_add)
            lang_ind += 1
            
    return res, transformer_indices


def get_target_lang_postags(model, dataset, data_loader, edit, mask_language, encoder_class):
    model.eval()
    res = {}
    data_endoer = encoder_class(data_loader, model, mask_language)
    
    if edit in dataset.nodes_map:
        with torch.no_grad():

            for z, verse, _, batch in data_endoer:
                if verse in dataset.nodes_map[edit]:

                    index = []
                    toks = list(dataset.nodes_map[edit][verse].keys())
                    for i in toks:
                        index.append(dataset.nodes_map[edit][verse][i])

                    # print(batch['x'][0][:, 10].shape)

                    batch['lang_based_nodes'], batch['transformer_indices'] = get_language_based_nodes(dataset.nodes_map, verse, index, batch['padding'][0], langs=[edit])
                    index = torch.LongTensor(index) - batch['padding'][0]

                    # print(torch.max(model.encoder.feature_encoder.layers[10].emb._parameters['weight'][batch['x'][0][index, :][:, 10].long(), :], dim=1))

                    preds = model.decoder(z, index, batch)

                    #print(verse)
                    #print(index.shape)
                    #print(index_.shape)
                    #print(preds.shape)
                    #print(z.shape)
                    #break

                    #if hasattr(model.decoder, 'transformer'):
                    #    #language_based_nodes = batch['lang_based_nodes'] # These two lines are for the transformer decoder, probably for normal decoder I have to delete the previouse lines
                    #    #preds = preds[language_based_nodes[0]]
                    #    preds = preds[index]

                    probabilities, predicted = torch.max(torch.softmax(preds, dim=1), 1)

                    res[verse] = [(toks[i], predicted[i].item(), probabilities[i].item()) for i in range(len(toks))]

    return res

def generate_target_lang_tags(model, target_lang, params, mask_language, train_dataset, grc_dataset, heb_dataset, blinker_dataset, train_dataloader, grc_dataloader, heb_dataloader, blinker_dataloader, encoder_class):
    target_pos_tags = {}
     
    res_ = get_target_lang_postags(model, train_dataset, train_dataloader, target_lang, mask_language, encoder_class)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, grc_dataset, grc_dataloader, target_lang, mask_language, encoder_class)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, heb_dataset, heb_dataloader, target_lang, mask_language, encoder_class)
    target_pos_tags.update(res_)

    res_ = get_target_lang_postags(model, blinker_dataset, blinker_dataloader, target_lang, mask_language, encoder_class)
    target_pos_tags.update(res_)

    torch.save(target_pos_tags, f'/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_{target_lang}_{params}_maskLang{mask_language}.pickle')
    return target_pos_tags



from copy import deepcopy
# These are functions to calculate the frequency vectors and also update training data with predictions for target language tags

# This funcion update existing structures that are used for training with new predictions
def update_trainset_with_predictions(node_cover, pos_labels, padding, index, max_values, pos_tags, word_added_to_train_freq, word_pos, target_train_treshold, tag_frequencies_max, tag_frequencies_max_value):
    pos_tags = pos_tags.to('cpu')
    index = index.to('cpu')
    pos_labels = pos_labels.to('cpu')
    #accepted_values = max_values > target_train_treshold
    #index = index[accepted_values]
    #pos_tags = pos_tags[accepted_values]
    #word_pos = word_pos[accepted_values]

    ## I should filter high repetition words here!
    #word_added_to_train_freq[word_pos.long()] += 1
    #accepted_by_frequency = word_added_to_train_freq[word_pos.long()] < 10
    #index = index[accepted_by_frequency]
    #pos_tags = pos_tags[accepted_by_frequency]
    #word_pos = word_pos[accepted_by_frequency]

    #if tag_frequencies_max != None:
    #    max_tag_number1 = tag_frequencies_max[0][word_pos.long()]
    #    max_tag_number2 = tag_frequencies_max[1][word_pos.long()]
    #    accepted_by_max_tag1 = torch.logical_and(max_tag_number1 == pos_tags, tag_frequencies_max_value[0][word_pos.long()] > 20.5)
    #    accepted_by_max_tag2 = torch.logical_and(torch.logical_and(max_tag_number2 == pos_tags, tag_frequencies_max_value[1][word_pos.long()] > 10.5), tag_frequencies_max_value[0][word_pos.long()] > 20.5)
    #    accepted_by_max_tag = torch.logical_or(accepted_by_max_tag1, accepted_by_max_tag2)
    #    pos_tags = pos_tags[accepted_by_max_tag]
    #    index = index[accepted_by_max_tag]

    if tag_frequencies_max != None:
        max_tag_number1 = tag_frequencies_max[0][word_pos.long()].to('cpu')
        max_tag_number2 = tag_frequencies_max[1][word_pos.long()].to('cpu')
        max_tag_number3 = tag_frequencies_max[2][word_pos.long()].to('cpu')
        accepted_by_max_tag1 = torch.logical_and(max_tag_number1 != pos_tags, tag_frequencies_max_value[0][word_pos.long()] > 8.5)
        accepted_by_max_tag2 = torch.logical_and(max_tag_number2 != pos_tags, tag_frequencies_max_value[1][word_pos.long()] > 5.5)
        accepted_by_max_tag3 = torch.logical_and(max_tag_number3 != pos_tags, tag_frequencies_max_value[2][word_pos.long()] > 3.5)
        accepted_by_max_tag = torch.logical_and(torch.logical_and(accepted_by_max_tag1, accepted_by_max_tag2), accepted_by_max_tag3)
        #accepted_by_max_tag = torch.logical_and(accepted_by_max_tag1, accepted_by_max_tag2)
        index = index[accepted_by_max_tag]
        max_tag_number1 = max_tag_number1[accepted_by_max_tag]

    index_global = index + padding
    node_cover.extend(index_global.tolist())
    pos_labels[index_global, pos_tags] = 1
    #pos_labels[index_global, max_tag_number1] = 1

# This function iterates a dataset to creates the tag vectors and update training structures
# for source languages call once
# for target twice. First to calculate tag frequencies for types. second time providing the calculated max_tag for each type
def get_words_tag_frequence(model, word_count, class_count, data_loader, encoder_class, tag_frequencies=None, from_gold_data=False, node_cover=None,
     pos_labels=None, mask_language=True, target_train_treshold=0.9, tag_frequencies_max=None, tag_frequencies_max_value=None):
    
    res = tag_frequencies
    if res == None:
        res = torch.ones(word_count, class_count)
        res[:, :] = 0.0000001
    
    data_encoder = encoder_class(data_loader, model, mask_language)
    word_added_to_train_freq = torch.zeros(word_count)

    with torch.no_grad():

        for z, verse, i, batch in data_encoder:
            index = batch['pos_index'][0]
            
            if from_gold_data:
                tags_onehot = batch['pos_classes'][0]
            else:
                tags_onehot = model.decoder(z, index, batch)
            #print(tags_onehot)
            #print(torch.softmax(tags_onehot, dim=1))
            #break
            max_values, pos_tags = torch.max(torch.softmax(tags_onehot, dim=1), 1)
            word_pos = batch['x'][0][index, 9]

            if not from_gold_data:
                if node_cover != None:
                    update_trainset_with_predictions(node_cover[verse], pos_labels, batch['padding'][0], index, max_values, pos_tags, word_added_to_train_freq, word_pos, target_train_treshold, tag_frequencies_max, tag_frequencies_max_value)
                accepted_values = max_values > 0.5
                word_pos = word_pos[accepted_values]
                pos_tags = pos_tags[accepted_values]
            
            res[word_pos.long(), pos_tags.long()] += 1



    sm = torch.sum(res, dim=1)
    res_normalized = (res.transpose(1,0) / sm).transpose(1,0)
    
    return res_normalized, res


# Calls the above functions over any of the datasets (train, grc, heb, blinker) to get pos_tag vectors for each word. (once for source languages and once for target languages)
def get_tag_frequencies_node_tags(model, editions, train_node_cover, train_labels, grc_node_cover, grc_labels, heb_node_cover, heb_labels, blinker_node_cover, blinker_labels, class_count,
                                    target_data_loader_train, target_data_loader_grc, target_data_loader_heb, target_data_loader_blinker,
                                    train_data_loader_bigbatch, grc_data_loader_bigbatch, heb_data_loader_bigbatch, blinker_data_loader_bigbatch, encoder_class, target_train_treshold, source_tag_frequencies=None,
                                    tag_frequencies_target=None, type_check=False):
    train_pos_node_cover_copy, train_pos_labels_copy = deepcopy(train_node_cover), deepcopy(train_labels)
    grc_pos_node_cover_copy, grc_pos_labels_copy = deepcopy(grc_node_cover), deepcopy(grc_labels)
    heb_pos_node_cover_copy, heb_pos_labels_copy = deepcopy(heb_node_cover), deepcopy(heb_labels)
    blinker_pos_node_cover_copy, blinker_pos_labels_copy = deepcopy(blinker_node_cover), deepcopy(blinker_labels)

    if source_tag_frequencies == None:
        _, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, train_data_loader_bigbatch, encoder_class, from_gold_data=True)
        get_words_tag_frequence(model, 2354770, class_count, grc_data_loader_bigbatch, encoder_class, from_gold_data=True, tag_frequencies=tag_frequencies)
        get_words_tag_frequence(model, 2354770, class_count, heb_data_loader_bigbatch, encoder_class, from_gold_data=True, tag_frequencies=tag_frequencies)
        get_words_tag_frequence(model, 2354770, class_count, blinker_data_loader_bigbatch, encoder_class, from_gold_data=True, tag_frequencies=tag_frequencies)
    else:
        tag_frequencies = source_tag_frequencies
    
    #_, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, blinker_data_loader_bigbatch, encoder_class, from_gold_data=True)

    if tag_frequencies_target == None:
        _, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_train, encoder_class, node_cover=None if type_check else train_pos_node_cover_copy, pos_labels=train_pos_labels_copy, target_train_treshold=target_train_treshold)
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_grc, encoder_class, node_cover=None if type_check else grc_pos_node_cover_copy, pos_labels=grc_pos_labels_copy, tag_frequencies=tag_frequencies_target, target_train_treshold=target_train_treshold)
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_heb, encoder_class, node_cover=None if type_check else heb_pos_node_cover_copy, pos_labels=heb_pos_labels_copy, tag_frequencies=tag_frequencies_target, target_train_treshold=target_train_treshold)
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, encoder_class, node_cover=None if type_check else blinker_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy, tag_frequencies=tag_frequencies_target, target_train_treshold=target_train_treshold)

    #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, encoder_class, node_cover=blinker_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy)

    if type_check:
        print('going to pick some training nodes')
        v, i = torch.max(tag_frequencies_target, dim=1)
        copy = tag_frequencies_target.detach().clone()
        first_dim = torch.arange(copy.size(0)).long()
        copy[first_dim, i] = 0.0000001
        v2, i2 = torch.max(copy, dim=1)
        copy[first_dim, i2] = 0.0000001
        v3, i3 = torch.max(copy, dim=1)

        _, t = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_train, encoder_class, node_cover=train_pos_node_cover_copy, pos_labels=train_pos_labels_copy, target_train_treshold=target_train_treshold, tag_frequencies_max=(i,i2,i3), tag_frequencies_max_value=(v, v2,v3))
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_grc, encoder_class, node_cover=grc_pos_node_cover_copy, pos_labels=grc_pos_labels_copy, target_train_treshold=target_train_treshold, tag_frequencies_max=(i,i2,i3), tag_frequencies_max_value=(v, v2, v3), tag_frequencies=t)
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_heb, encoder_class, node_cover=heb_pos_node_cover_copy, pos_labels=heb_pos_labels_copy, target_train_treshold=target_train_treshold, tag_frequencies_max=(i,i2,i3), tag_frequencies_max_value=(v, v2, v3), tag_frequencies=t)
        get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, encoder_class, node_cover=blinker_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy, target_train_treshold=target_train_treshold, tag_frequencies_max=(i,i2,i3), tag_frequencies_max_value=(v, v2,v3), tag_frequencies=t)
    
    tag_frequencies += tag_frequencies_target.to(torch.device('cpu'))
        
    return tag_frequencies, tag_frequencies_target, train_pos_node_cover_copy, train_pos_labels_copy, grc_pos_node_cover_copy, grc_pos_labels_copy, heb_pos_node_cover_copy, heb_pos_labels_copy, blinker_pos_node_cover_copy, blinker_pos_labels_copy


def keep_only_type_tags(tag_frequencies_copy):
    bool_tensor = torch.zeros(tag_frequencies_copy.shape).bool()
    bool_tensor[:, :] = True

    _, max1 = torch.max(tag_frequencies_copy, dim=1)
    copy = tag_frequencies_copy.detach().clone()
    first_dim = torch.arange(copy.size(0)).long()
    copy[first_dim, max1] = 0.0000001
    _, max2 = torch.max(copy, dim=1)
    # copy[first_dim, max2] = 0.0000001
    # _, max3 = torch.max(copy, dim=1)

    bool_tensor[first_dim, max1] = False
    bool_tensor[first_dim, max2] = False
    # bool_tensor[first_dim, max3] = False
    tag_frequencies_copy[bool_tensor] = 0.0000001
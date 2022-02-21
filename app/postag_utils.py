import torch

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
                    index = torch.LongTensor(index) - batch['padding'][0]

                    preds = model.decoder(z, index, batch)

                    _, predicted = torch.max(preds, 1)

                    res[verse] = {toks[i]:predicted[i].item() for i in range(len(toks))}

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
def update_trainset_with_predictions(node_cover, pos_labels, padding, index, max_values, pos_tags, word_added_to_train_freq, word_pos):
    
    accepted_values = max_values > 0.6
    index = index[accepted_values]
    pos_tags = pos_tags[accepted_values]
    word_pos = word_pos[accepted_values]

    # I should filter high repetition words here!
    #word_added_to_train_freq[word_pos.long()] += 1
    #accepted_by_frequency = word_added_to_train_freq[word_pos.long()] < 10
    #index = index[accepted_by_frequency]
    #pos_tags = pos_tags[accepted_by_frequency]

    index_global = index + padding
    node_cover.extend(index_global.tolist())
    pos_labels[index_global, pos_tags] = 1

# This function iterates a dataset to creates the tag vectors and update training structures
def get_words_tag_frequence(model, word_count, class_count, data_loader, encoder_class, tag_frequencies=None, from_gold_data=False, node_cover=None, pos_labels=None, mask_language=True):
    
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
                    update_trainset_with_predictions(node_cover[verse], pos_labels, batch['padding'][0], index, max_values, pos_tags, word_added_to_train_freq, word_pos)
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
                                    train_data_loader_bigbatch, grc_data_loader_bigbatch, heb_data_loader_bigbatch, blinker_data_loader_bigbatch, encoder_class):
    train_pos_node_cover_copy, train_pos_labels_copy = deepcopy(train_node_cover), deepcopy(train_labels)
    grc_pos_node_cover_copy, grc_pos_labels_copy = deepcopy(grc_node_cover), deepcopy(grc_labels)
    heb_pos_node_cover_copy, heb_pos_labels_copy = deepcopy(heb_node_cover), deepcopy(heb_labels)
    blinker_pos_node_cover_copy, blinker_pos_labels_copy = deepcopy(blinker_node_cover), deepcopy(blinker_labels)

    #_, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, train_data_loader_bigbatch, from_gold_data=True)
    #_, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, grc_data_loader_bigbatch, from_gold_data=True, tag_frequencies=tag_frequencies)
    #_, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, heb_data_loader_bigbatch, from_gold_data=True, tag_frequencies=tag_frequencies)
    #_, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, blinker_data_loader_bigbatch, from_gold_data=True, tag_frequencies=tag_frequencies)

    _, tag_frequencies = get_words_tag_frequence(model, 2354770, class_count, blinker_data_loader_bigbatch, encoder_class, from_gold_data=True)

    

    #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_train, node_cover=train_pos_node_cover_copy, pos_labels=train_pos_labels_copy)
    #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_grc, node_cover=grc_pos_node_cover_copy, pos_labels=grc_pos_labels_copy, tag_frequencies=tag_frequencies_target)
    #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_heb, node_cover=heb_pos_node_cover_copy, pos_labels=heb_pos_labels_copy, tag_frequencies=tag_frequencies_target)
    #_, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, node_cover=blinker_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy, tag_frequencies=tag_frequencies_target)

    _, tag_frequencies_target = get_words_tag_frequence(model, 2354770, class_count, target_data_loader_blinker, encoder_class, node_cover=blinker_pos_node_cover_copy, pos_labels=blinker_pos_labels_copy)

    tag_frequencies += tag_frequencies_target
    return tag_frequencies, tag_frequencies_target, train_pos_node_cover_copy, train_pos_labels_copy, grc_pos_node_cover_copy, grc_pos_labels_copy, heb_pos_node_cover_copy, heb_pos_labels_copy, blinker_pos_node_cover_copy, blinker_pos_labels_copy

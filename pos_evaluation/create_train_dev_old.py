"""
Create train and dev set from bronze data

Example call:
$ python3 create_train_dev.py --pos /mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_tam-x-bible-newworld_posfeatFalse_transformerFalse_trainWEFalse_maskLangTrue.pickle --bible tam-x-bible-newworld.txt --bronze 1 --lang tam
$ python3 create_train_dev.py --pos /mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_fin-x-bible-helfi_posfeatFalse_transformerFalse_trainWEFalse_maskLangTrue.pickle --bible fin-x-bible-helfi.txt --bronze 1 --lang fin

"""

import torch
import random
import argparse

def load_bible(filename):
    bible = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            l = line.strip().split("\t")
            if len(l)<2:
                continue
            bible[l[0]] = l[1]   
    print("Len bible "+filename+" = "+str(len(bible)))
    # print(bible)
    return bible

def get_sets(pos_tags_file, bible_file, bronze, lang):
    postag_map = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, 
                "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
    inv_postag_map = {v: k for k, v in postag_map.items()}

    # pos_tags = torch.load('/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_freeze-embedding_noLang.pickle')
    # bronze 1
    # pos_tags = torch.load('/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_posfeatFalse_transformerFalse_trainWEFalse_maskLangTrue.pickle')
    # bronze 2
    # pos_tags = torch.load('/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_posfeatTrue_transformerFalse_trainWEFalse_maskLangTrue.pickle')
    # bronze 3
    pos_tags = torch.load(pos_tags_file)
    # bible = load_bible("/nfs/datc/pbc/"+bible_file)
    bible = load_bible("/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/"+bible_file)
    # train = open("yoruba_silver_train.txt", "w+")
    # dev = open("yoruba_a_silver_train.txt", "w+")
    # train = open("yoruba_bronze1_train.txt", "w+")
    # dev = open("yoruba_bronze1_dev.txt", "w+")
    train = open(lang+"_bronze"+bronze+"_train.txt", "w+")
    dev = open(lang+"_bronze"+bronze+"_dev.txt", "w+")
    ids_file = open(lang+"_ids.txt", "w+")            ### To Fix: USE SAME IDS 

    print(len(pos_tags))
    # shuffle pos_tags file and take 80% train, 10% dev
    ids = list(pos_tags.keys())
    random.shuffle(ids)
    n_train = int(len(ids)*90/100)
    print("Len dev: ", len(ids)-n_train)
    ids = ids[0:n_train]
    
    print("Len train: ", n_train)

    for verse_id in pos_tags:
        # print(verse_id)
        if verse_id in ids:
            flag = 1
            ids_file.write(verse_id+"\n")
        else:
            flag = 0

        bible_verse = bible[verse_id].split(" ")

        # get pairs word-tag
        for i in range(len(bible_verse)):
            if i in pos_tags[verse_id]:
                if flag:
                    train.write(verse_id+"\t"+bible_verse[i]+"\t"+inv_postag_map[pos_tags[verse_id][i]]+"\n")
                else:
                    dev.write(verse_id+"\t"+bible_verse[i]+"\t"+inv_postag_map[pos_tags[verse_id][i]]+"\n")
            else:
                if flag:
                    train.write(verse_id+"\t"+bible_verse[i]+"\t"+"X"+"\n")
                else:
                    dev.write(verse_id+"\t"+bible_verse[i]+"\t"+"X"+"\n")

        if flag: # add space in between sentences
            train.write("\n")
        else:
            dev.write("\n")

    train.close()
    dev.close()
    ids_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", default=None, type=str, required=True, help="Bronze pickle")
    parser.add_argument("--bible", default=None, type=str, required=True, help="Bible file")
    parser.add_argument("--bronze", default=None, type=int, required=True, help="Specify bronze number [1,2,3]")
    parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
    args = parser.parse_args()

    pos_tags_file = args.pos
    bible_file = args.bible
    bronze = str(args.bronze)
    lang = args.lang
    get_sets(pos_tags_file, bible_file, bronze, lang)


if __name__ == "__main__":
    main()



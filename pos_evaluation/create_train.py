"""

Create train from bronze data
Example call:
$ python3 create_train.py --bronze_file /mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_hin-x-bible-bsi_15lngs-POSFeatTruealltgts_trnsfrmrTrue6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.8_TypchckTrue_TgBsdSlctTrue_tstamtall_20220502-155236_ElyStpDlta0-GA-chnls1024_small.pickle
                          --bronze 3
                          --bible hin-x-bible-bsi.txt 
                          --lang hin
                          --thr 0.8
"""


import collections
from time import strftime
import torch
from math import log, log10
import codecs
# Statistics
import re
import argparse


postag_map = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, 
"PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16, "***": 17}
postag_reverse_map = {item[1]:item[0] for item in postag_map.items()}
# source_distr = torch.load('/mounts/data/proj/ayyoob/POS_tagging/dataset/small/distributions.torch.bin')
source_distr = torch.load('/mounts/data/proj/ayyoob/POS_tagging/dataset/small_final/distributions_all.torch.bin')
# Checking for trivial POS types
# REGEX for numbers, punctuation marks and symbols
REGEX_DIGIT = '[\d٠١٢٣٤٥٦٧٨٩౦౧౨౩౪౫౬౭౮౯፲፳፴፵፶፷፸፹፺፻०१२३४५६७८९४零一二三四五六七八九十百千万億兆つ]'
REGEX_PUNCT1 = r'^[\॥\\\\_\"\“\”\‘\’\``\′\՛\·\.\ㆍ\•\,\,\、\;\:\?\？\!\[\]\{\}\(\)\|\«\»\…\،\؛\؟\¿\፤\፣\።\፨\፠\፧\፦\፡\…\।\¡\「\」\《\》\』\『\‹\〔\〕\–\—\−\-\„\‚\´\'\〉\〈 \【\】\（\）\~\。\○\．\♪]+$'
REGEX_PUNCT2 = r'^[\*\/\-]{2,}$'
REGEX_SYM1 = r'^[\+\=\≠\%\$\£\€\#\°\@\٪\≤\≥\^\φ\θ\×\✓\✔\△\©\☺\♥\❤]+$'
REGEX_SYM2 = r'^((\:[\)\(DPO])|(\;[\)])|m²)$'
REGEX_SYM3 = r'^'+REGEX_DIGIT+'+(([\.\,\:\-\/])?'+REGEX_DIGIT+')*\%$'
REGEX_NUMBER = r'^\%?'+REGEX_DIGIT+'+(([\.\,\:\-\/])?'+REGEX_DIGIT+')*$'
REGEX_EMOJI = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "]+)"
)

def is_number(token):
    return re.match(REGEX_NUMBER, token)

def is_punctuation(token):
    return re.match(REGEX_PUNCT1, token) or re.match(REGEX_PUNCT2, token)

def is_symbol(token):
    return re.match(REGEX_SYM1, token) or re.match(REGEX_SYM2, token) or re.match(REGEX_SYM3, token) or re.match(REGEX_EMOJI, token)

def read_bible_file(f_name, sentences, type_counts):
    with codecs.open('/nfs/datc/pbc/'+f_name, 'r', 'utf-8') as fi:
    # with codecs.open('/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/'+f_name, 'r', 'utf-8') as fi:
        for l in fi:
            l = l.strip()
            lparts = l.split('\t')
            if l.startswith('#') or l == '' or len(lparts)<2:
                continue
            
            toks = lparts[1].split()
            sentences[lparts[0]] = toks
            for tok in toks:
                type_counts[tok] +=1

def read_our_bronze_file_source(f_path, bible_file_name, threshold, lang):

    sentences = {}
    type_counts = collections.defaultdict(lambda:0)  # number of times a token happened
    type_tag_counts = collections.defaultdict(lambda: collections.defaultdict(lambda:0))
    type_tag_probabilities = collections.defaultdict(lambda: collections.defaultdict(lambda:0.0))
    read_bible_file(bible_file_name, sentences, type_counts)
    data = torch.load(f_path)
    max_count, max_degree = 0, 0
    res = {}
    res_out = {}

    for sent in data:
        for item in data[sent]:
            type = sentences[sent][item[0]] # word
            type_tag_counts[type][item[1]] += 1 # count how many times a word get a certain tag (item[1])
            type_tag_probabilities[type][item[1]] += 1
            max_degree = max(max_degree, item[3])

    for type in type_tag_probabilities:
        max_count = max(max_count, type_counts[type])
        for tag in type_tag_probabilities[type]:
            type_tag_probabilities[type][tag] /= type_counts[type]

    source = {}
    for word in source_distr[lang]:
        source[word] = [ int(x) for x in source_distr[lang][word]]
        # if max(source[word]) == 0:
        #     print(word, source[word])
        max_value = max(1,max(source[word]))
        source[word] = [x/max_value for x in source[word]]

        # sum_value = max(1,sum(source[word]))
        # source[word] = [x/sum_value for x in source[word]]

    count_sent = 0
    for sent in data:

        count_sent+=1
        res[sent] = {}
        res_out[sent] = {}
        for item in data[sent]:
            
            type = sentences[sent][item[0]]
            try:
                p_type_tag = source[type][item[1]] # distribution of tags
            except:
                print("Missing: ", item)
                p_type_tag = 1

            # strong constraint
            # if p_type_tag <= 0.3:
            #     p_type_tag = 0

            p_type_target = type_tag_probabilities[type][item[1]]

            # p_type_tag = (p_type_tag+type_tag_probabilities[type][item[1]])/2 # avg source and target distributions

            p_mul = item[2] * p_type_tag # item[2]=probability given by gnn
            # p_sum = item[2] + p_type_tag
            # size_factor = log10(type_counts[type])  
            size_factor = 1+(log(type_counts[type]+1, 10)/log(max_count+1, 10)) # normalized number of times a token happens
            # size_factor = type_counts[type]/max_count # normalized number of times a token happens
            # if size_factor * p_mul > threshold:
            if p_mul > threshold: # or item[2]>0.9:
            # if size_factor * item[2] > threshold:
            # if size_factor * p_sum > threshold:
            # if item[2]>threshold:
            # if size_factor * p_mul * degree > threshold:
            # if p_type_tag:
            # if size_factor * p_mul * p_type_target > threshold:
            # if (size_factor * item[2] * max(p_type_tag,p_type_target) > threshold): # or (p_type_target==1 or p_type_tag==1):
            # if (size_factor * item[2] * min(p_type_tag,p_type_target) > threshold): # or (p_type_target==1 or p_type_tag==1):
            # if (size_factor * item[2] * (p_type_tag+p_type_target)/2 > threshold): # or (p_type_target==1 or p_type_tag==1):
            # if item[2] * p_type_tag * p_type_target > threshold: # or (p_type_target==1 or p_type_tag==1):
                res[sent][item[0]] = postag_reverse_map[item[1]]
                # if sent in sentences and item[1]!=17:
                # res_out[sent][sentences[sent][item[0]]] = postag_reverse_map[item[1]]
                res_out[sent][item[0]] = postag_reverse_map[item[1]]

                # if sent=='02007019':
                #     print(item[0],res_out[sent][sentences[sent][item[0]]])
                # else:
                #     print("error")

            else:
                # res_out[sent][sentences[sent][item[0]]] = "***"
                res_out[sent][item[0]] = "***"

                # if count_sent>25 and count_sent<28 and (size_factor * p_mul)>0:  
                #     print(sentences[sent][item[0]], item[0], postag_reverse_map[item[1]], size_factor * p_mul)
                #     print("Size factor: ", size_factor)
                #     print("P_mul ", p_mul)
                #     print("p source ", p_type_tag)
                #     print("p_target ", p_type_target)
                #     print("item ", item[2])
                #     # print("mul: ", size_factor * p_mul * p_type_target)
                #     # print("mul, max: ", size_factor * item[2] * max(p_type_tag,p_type_target))
                #     # print("mul, avg: ", size_factor * item[2] * (p_type_tag+p_type_target)/2)
                #     print("mul: ", item[2] * p_type_tag * p_type_target)

                #     print("\n")

    return res, res_out

def struct_to_file(structure_out, bible_file_name, out_file, density=0):
    sentences = {}
    type_counts = collections.defaultdict(lambda:0)  # number of times a token happened

    read_bible_file(bible_file_name, sentences, type_counts)
    count_star = 0
    tot = 0
    toolong = 0
    bad_sentences = 0
    with open(out_file, "w+") as outfile:
        for sent in structure_out:
            if len(structure_out[sent])>80:
                toolong+=1
                continue
            partial = 0
            if density:
                for word_id in range(len(structure_out[sent])): 
                    
                    if structure_out[sent][word_id] == "***":
                        partial+=1
                if 1-(partial/len(structure_out[sent]))<density:#0.5:
                # if 1-(partial/len(structure_out[sent]))<0:
                    # print(partial, len(structure_out[sent]))
                    bad_sentences+=1
                    continue
                
            outfile.write("# verse_id: "+ sent + "\n")
            # for word_id in structure_out[sent]:
            for word_id in range(len(structure_out[sent])): 
                tot += 1
                # print(sentences[sent])
                token = sentences[sent][word_id]
                label = structure_out[sent][word_id]
                
                if not is_symbol(token) and label=='SYM':
                    label = '***'
                elif not is_punctuation(token) and label=='PUNCT':
                    # print(token)
                    if len(token)==1 and ord(token) == 2405:
                        continue
                    label = '***'
                # elif not is_number(token) and label=='NUM':
                #     label = '***'

                if  is_symbol(token) and not label=='SYM':
                    label = 'SYM'
                elif is_punctuation(token) and not label=='PUNCT':
                    label='PUNCT'
                elif is_number(token) and not label=='NUM':
                    label='NUM'
                
                # Reduce sparsity by replacing all digits with 0.
                # if token.isdigit() or is_number(token) or (is_number(token) and is_number(token)) or structure_out[sent][word] == "NUM":
                #     token = '0'
                

                outfile.write(sent+"\t"+token+"\t"+token.lower()+"\t"+label+"\n")
                if label == "***":
                    count_star+=1
            
                
        
            outfile.write("\n")
    print(f"Null tokens: {count_star/tot}")
    print(f"Removed sentences because of density below {density}: {bad_sentences}")
    print(f"Too long sentences: {toolong}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", default=None, type=int, required=True, help="Specify bronze number [1,2,3]")
    parser.add_argument("--bronze_file", default=None, type=str, required=True, help="Specify bronze file")
    parser.add_argument("--bible", default=None, type=str, required=True, help="Specify bible file")
    parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
    parser.add_argument("--thr", default=None, type=float, required=True, help="Threshold for filtering")
    parser.add_argument("--base_path", default=None, type=str, required=False, help="Base path")
    args = parser.parse_args()

    bronze, bronze_out = read_our_bronze_file_source(args.bronze_file, args.bible, args.thr, args.lang) 
    base_path = "/mounts/work/silvia/POS/filter/"
    if args.base_path:
        base_path = args.base_path
    print(f"Output file: {base_path+args.lang}_bronze{args.bronze}_{str(args.thr)}.conllu")
    struct_to_file(bronze_out, args.bible, base_path+args.lang+"_bronze"+str(args.bronze)+"_"+str(args.thr)+".conllu", density=0)

if __name__ == "__main__":
    main()
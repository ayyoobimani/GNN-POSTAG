"""

Create brown clusters embeddings

Source:
- bible
- wikipedia dumps

Steps:
1- download file
2- python -m wikiextractor.WikiExtractor --json gvwiki-20201220-pages-articles-multistream.xml.bz2
3- python3 brown_clusters.py --wiki_folder /mounts/work/silvia/wikipedia_dumps/text --wiki_file glv_wiki.txt --bible_file glv-x-bible.txt  --test gv_cadhan-ud-test_2_7.conllu


"""

import sys
import os
from datetime import datetime
import re
import torch
import gensim
import glob
import json
import argparse
from blingfire import text_to_sentences
import subprocess

def blingfire_run(wiki_dump_folder_in, wiki_dump_file_out):
    # https://yulianudelman.medium.com/build-a-corpus-for-nlp-models-from-wikipedia-dump-file-475b21145885
    print("Blingfire:\n")
    wiki_dump_folder_in=f'{wiki_dump_folder_in}/**/*'
    with open(wiki_dump_file_out, 'w', encoding='utf-8') as out_f:
        for filename in glob.glob(wiki_dump_folder_in):
            filename=filename.replace("\\","/")
            articles = []
            for line in open(filename, 'r'):
                articles.append(json.loads(line))
                for article in articles:
                    sentences = text_to_sentences(article['text'])
                    out_f.write(sentences + '\n')

REGEX_PUNCT1 = r'^[\॥\\\\_\"\"\“\”\‘\’\``\′\՛\·\.\ㆍ\•\,\,\、\;\:\?\？\!\[\]\{\}\(\)\|\«\»\…\،\؛\؟\¿\፤\፣\።\፨\፠\፧\፦\፡\…\।\¡\「\」\《\》\』\『\‹\〔\〕\–\—\−\-\„\‚\´\'\〉\〈 \【\】\（\）\~\。\○\．\♪]+$'
REGEX_PUNCT2 = r'^[\*\/\-]{2,}$'
def is_punctuation(token):
    return re.match(REGEX_PUNCT1, token) or re.match(REGEX_PUNCT2, token)

def tokenize(sentence):
    out = ""
    for char in sentence.strip():
        if char==" ":
            out+=char
        elif not is_punctuation(char):
            out+=char
        else:
            out+=" "+char
    return out

def get_files_for_clustering(bible, wiki_file):
    print("\n Get words from clustering:\n")
    bible_file = "/nfs/datc/pbc/"+bible
    out_file = open("brown_sentences_"+bible,"w+")
    wiki_out = open("brown_sentences_"+wiki_file,"w+")
    uniq_words = {}

    with open(bible_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            l = line.strip().split("\t")
            if len(l)<2:
                continue
            out_file.write(l[1]+"\n")
            for word in l[1].split(" "):
                uniq_words[word] = 1

    # # tokenize wiki file
    tot = 0
    interval = 0
    flag = 0

    with open(wiki_file) as f:
        for line in f:
            interval += 1
            # if len(uniq_words.keys())>500000:
            if len(uniq_words.keys())>5000:
                break
            if interval%10000==0:
                print(len(uniq_words.keys()), tot)

            try:
                tokenized = tokenize(line)
                tot+=len(tokenized.split(" "))
                wiki_out.write(tokenized)
                wiki_out.write("\n")
                for word in tokenized.split(" "):
                    uniq_words[word] = 1
            except:
                print(line)

    out_file.close()
    wiki_out.close()
    print("tot words: ", tot)
    print("Unique words: ", len(uniq_words))

def clustering(bible_file, wiki_file):
    output_folder = f"{bible_file}_{wiki_file}-c128-p1.out"
    command = f"cat brown_sentences_{bible_file} brown_sentences_{wiki_file} | /mounts/work/silvia/POS/brown-cluster/wcluster --text /dev/stdin --c 128 --output_dir {output_folder}"
    print(command)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start time =", dt_string)	
    os.system(command)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End time =", dt_string)	

"""
code from baseline
"""
BROWN_SIZE = 16
START_MARKER = '<START>'
END_MARKER = '<END>'
class BROWN_Processor:
    
    def __init__(self, brown_clusters_path, brown_size):

        self.brown_size = brown_size
        self.cluster_map = {}
        self.string_map = {}
        self.max_brown_cluster_index = 0
        self.cluster_n_map = [None]*(self.brown_size)
        self.unknown_map = {}

        index = 0
        string_map = {}
        
        for i in range(self.brown_size):
             self.cluster_n_map[i] = {}

        count = 1

        # Construct the clusters given an output file of words and their clusters.
        if brown_clusters_path != None:
            for line in open(brown_clusters_path):
                if len(line) > 0:
                    count += 1
                    line_data = line.split()
                    if len(line_data) > 1:
                        cluster = line_data[0]
                        word = line_data[1]
                        # word = simplify_token(word)
                        cluster_num = index  # counter variable
                        
                        if cluster not in string_map:
                            self.cluster_map[word] = index
                            string_map[cluster] = index
                            index += 1
                        else:
                            cluster_num = string_map[cluster]       # give same numbers (index) to same clusters
                            self.cluster_map[word] = cluster_num    # save number (prev index) for a give word
                        for i in range(self.brown_size):            # brown_size = 16
                            pref_id = index;
                            prefix = cluster[0: min(i+1, len(cluster))] # compute all prefix of cluster (which are suffixes in the end)
                                                                        # ancestors = the prefixes of the BINARY representation
                            if prefix not in string_map:
                                string_map[prefix] = index  # add new cluster to string_map if it doesn't exist
                                index += 1
                            else:
                                pref_id = string_map[prefix]

                            self.cluster_n_map[i][cluster_num] = pref_id

                if index > self.max_brown_cluster_index:
                    self.max_brown_cluster_index = index

        print("Brown Max: ", self.max_brown_cluster_index) # final number of clusters, included ancestors (prefixes)

    # Retrieve the clusters of a given word.
    def get_brown_cluster(self, word):
        output = []
        if word == START_MARKER:
            return [-1]*(10*(self.brown_size+1))
        if word == END_MARKER:
            return [1]*(10*(self.brown_size+1))
        # word = simplify_token(word)
        ids  = [None]*(self.brown_size+1) # size 17
        
        # initialize with 0
        for i in range(0, self.brown_size+1): # 17
            ids[i] = 0

        if word in self.cluster_map:
             ids[0] = self.cluster_map[word]  # assign cluster id to ids[0]
            #  print(self.cluster_map[word])
             
        if ids[0] > 0:
            for j in range(1, self.brown_size+1):
                ids[j] = self.cluster_n_map[j-1][ids[0]]  # prefix cluster number

        for j in range(len(ids)):
            # print(ids[j], bin(ids[j])[2:])  
            out1 = [int(i) for i in bin(ids[j])[2:]] # convert cluster number to binary
            out2 = [0]* (10 - len(out1)) + out1  # bring each binary number to 10 digits
            output.extend(out2) # the result's length is 170
        return output

def load_brown(bible_file, wiki_file):
    print("Load brown clusters")
    return BROWN_Processor(f"{bible_file}_{wiki_file}-c128-p1.out/paths", BROWN_SIZE)

#Prepare embedding.vec file for brown clusters for all sentences in train and test
def save_clusters_170(bible_file, wiki_file):
    print("Convert brown clustering file to word2vec format. Pad to 170.")

    brown_processor = load_brown(bible_file, wiki_file)
    saved = {}
    cluster_file = f"{bible_file}_{wiki_file}-c128-p1.out/paths"
    vector_file = f"brown_clusters_170_{bible_file}_{wiki_file}.vec"
    out = open(vector_file, "w+")

    with open(cluster_file) as f:
        for line in f:
            l = line.strip().split("\t")
            word = l[1]
            if word in saved:
                continue
            saved[word] = 1
            embedding = brown_processor.get_brown_cluster(word)
            if len(embedding)!=170:
                print(word)
                continue
            # print(embedding)
            out.write(word)
            for element in embedding:
                out.write(" "+str(element))
            out.write("\n")

    # add length and size at the top of vector file
    command = f"wc -l {vector_file}"
    length_file = subprocess.getoutput(command).split()[0]
  
    command = f" sed -i '1s/^/{length_file} 170\\n/' {vector_file}"
    print(command)
    os.system(command)

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False)
    word_vectors.save(vector_file+'.gensim')

#Convert brown clustering file to word2vec format. Pad to 16.
def save_clusters_16(bible_file, wiki_file):
    print("Convert brown clustering file to word2vec format. Pad to 16.")
    size = 0
    max_len = 0
    cluster_file = f"{bible_file}_{wiki_file}-c128-p1.out/paths"
    vector_file = f"{bible_file}_{wiki_file}-c128-p1.out/paths.vec"
    out = open(vector_file,"w+")
    with open(cluster_file) as f:
        for line in f:
            l = line.strip().split("\t")
            max_len = max(max_len,len(l[0]))
            emb = [0]* (16 - len(l[0])) + list(l[0])
            word = l[1]
            out.write(word)
            for el in emb:
                out.write(" "+str(el))
            out.write("\n")
            size+=1

    out.close()
    # print(size)
    # print(max_len)

     # add length and size at the top of vector file
    command = f"wc -l {vector_file}"
    length_file = subprocess.getoutput(command).split()[0]
  
    command = f" sed -i '1s/^/{length_file} 16\\n/' {vector_file}"
    print(command)
    os.system(command)

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(f"{bible_file}_{wiki_file}-c128-p1.out/paths.vec", binary=False)
    word_vectors.save(f"{bible_file}_{wiki_file}-c128-p1.out/paths.vec"+'.gensim')

"""
Count if test covered
"""
def is_test_covered(bible_file, wiki_file, test_set):
    print("\nIs test covered?")
    vector_file = f"brown_clusters_{bible_file}_{wiki_file}.vec"
    words = {}
    in_words, out_words = 0, 0

    with open(vector_file) as f:
        for line in f:
            l = line.strip().split()
            words[l[0]] = 1

    tags = {}
    with open("/nfs/datx/UD/v2_5/"+test_set) as f:
        for line in f:
            if line.startswith("#") or line=="\n":
                continue
            l = line.strip().split("\t")
            if l[1] in words:
                in_words+=1
            else:
                out_words+=1
                if l[3] in tags:
                    tags[l[3]]+=1
                else:
                    tags[l[3]] = 1

    print("Covered words: ",in_words)
    print("Not-covered: ", out_words)
    print(tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_file", default=None, type=str, required=False, help="Wikipedia dump")
    parser.add_argument("--bible_file", default=None, type=str, required=False, help="Bible file")
    parser.add_argument("--wiki_folder", default=None, type=str, required=False, help="Wiki dump zip")
    parser.add_argument("--test", default=None, type=str, required=False, help="Test file")
    args = parser.parse_args()

    blingfire_run(args.wiki_folder, args.wiki_file)
    get_files_for_clustering(args.bible_file, args.wiki_file)
    clustering(args.bible_file, args.wiki_file)
    save_clusters_170(args.bible_file, args.wiki_file)
    save_clusters_16(args.bible_file, args.wiki_file)
    is_test_covered(args.bible_file, args.wiki_file, args.test)

if __name__ == "__main__":
    main()

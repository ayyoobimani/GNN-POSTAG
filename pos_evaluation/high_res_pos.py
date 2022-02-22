"""
Tag high-resource languages with Flair POS tagger

Input:
--list : file with bible editions list
--gpu : gpu number
--ouput :  path to save output tagged files

Ouput:
- one file per language in conllu format save in "--output" path 

Example:
$ python3 high_res_pos.py --list editions_to_pos_tag.txt --gpu 5 --/mounts/work/silvia/POS/TAGGED_LANGS/
"""

import codecs
from email.policy import default
from tqdm import tqdm
import torch
import flair
from flair.models import SequenceTagger
from flair.data import Sentence
import argparse
import os

def tag_languages(lang_list_file, output_tag_path):

    pbc_path = "/nfs/datc/pbc/"
    helfi_path = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/"

    device = torch.device('cuda')
    flair.device = device

    tagger = {}
    # tagger["eng"] = SequenceTagger.load('upos')
    tagger["all"] = SequenceTagger.load('flair/upos-multi')

    with open(lang_list_file) as f:
        for line in f:
            edition = line.strip()
            print(edition, "...")
            if edition != "fin-x-bible-helfi":
                file_in = codecs.open(F"{pbc_path}{edition}.txt", "r", "utf-8")
            else:
                file_in = codecs.open(F"{helfi_path}{edition}.txt", "r", "utf-8")
            
            file_out = codecs.open(F"{output_tag_path}/{edition}.conllu", "w", "utf-8")

            ds = [l.strip() for l in file_in if l.strip() != ""]
            data_loader = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False)

            sent_id = 0
            for id_batch, l_batch in enumerate(tqdm(data_loader)):
                ll_batch = [Sentence(l.split("\t")[1]) for l in l_batch if len(l.split("\t")) == 2 and l[0] != "#"]
                verse_batch = [l.split("\t")[0] for l in l_batch if len(l.split("\t")) == 2 and l[0] != "#"]

                # Tag line
                if edition[:3] in tagger:
                    tagger[edition[:3]].predict(ll_batch)
                else:
                    tagger["all"].predict(ll_batch)

                # Write in conllu format
                for vs, ll in zip(verse_batch, ll_batch):
                    tagged_l = ll.to_tagged_string().strip().split()
                    l_tags = [[tagged_l[i], tagged_l[i+1][1:-1]] for i in range(len(tagged_l)) if i % 2 == 0]

                    file_out.write(F"# verse_id = {vs}\n")
                    file_out.write(F"# sent_id = {sent_id}\n")
                    file_out.write(F"# text = " + " ".join([w_p[0] for w_p in l_tags]) + "\n")
                    for i, w_p in enumerate(l_tags):
                        line_str = F"{i+1}\t{w_p[0]}\t{w_p[0]}\t{w_p[1]}\t_\t_\t0\t_\t_\t_\n"
                        file_out.write(line_str)
                    file_out.write("\n")
                    sent_id += 1

            file_in.close()
            file_out.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default=None, type=str, required=True, help="Bible editions list file")
    parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
    parser.add_argument("--output", default="/mounts/work/silvia/POS/TAGGED_LANGS/", type=str, help="Output path")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    tag_languages(args.list, args.output)

if __name__ == "__main__":
    main()

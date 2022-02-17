"""
POS tagger with BERT embeddings

$ python3 pos_tagger_bert.py --bronze 1 --lang tam --fasttext cc.ta.300.vec --gpu 5

"""
from flair.datasets import UD_ENGLISH, UD_FINNISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, FastTextEmbeddings, TransformerWordEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import os
import gensim
import argparse

def train(bronze, lang, fasttext):

    # 1. get the corpus
    columns = {1: 'text', 2: 'upos'}

    # this is the folder in which train, test and dev files reside
    data_folder = '/mounts/work/silvia/POS'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file =   lang+'_'+bronze+'_train.txt',
                                test_file  =   lang+'_'+bronze+'_dev.txt',
                                dev_file   =   lang+'_'+bronze+'_dev.txt',
                                )

    # 2. what label do we want to predict?
    label_type = 'upos'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    # 4. initialize embeddings
    embedding_types = [
        TransformerWordEmbeddings('bert-base-multilingual-cased')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=128,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/'+lang+'-upos-bert-'+bronze,
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=50,
                checkpoint=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", default=None, type=int, required=True, help="Specify bronze number [1,2,3]")
    parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
    parser.add_argument("--fasttext", default=None, type=str, required=True, help="Fasttext file")
    parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    bronze = "bronze"+str(args.bronze)
    lang = args.lang
    fasttext = args.fasttext
    train(bronze, lang, fasttext)


if __name__ == "__main__":
    main()


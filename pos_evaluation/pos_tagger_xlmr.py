"""
POS tagger with XLMR embeddings

$ python3 pos_tagger_xlmr.py --bronze 1 --lang tam --gpu 5 --train --test

Eflomal:
$ cat hin_eng_eflomal_gdfa_dev.conllu > hin_eng_eflomal_gdfa_all.conllu
$ cat hin_eng_eflomal_gdfa_train.conllu >> hin_eng_eflomal_gdfa_all.conllu
$ python3 pos_tagger_xlmr.py --bronze eflomal --lang hin --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/hin_eng_eflomal_gdfa_all.conllu --test hi_hdtb-ud-test_2_5.conllu
$ python3 pos_tagger_xlmr.py --bronze eflomal --lang por --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/por_eng_eflomal_gdfa_all.conllu --test pt_bosque-ud-test_2_5.conllu
$ python3 pos_tagger_xlmr.py --bronze eflomal --lang ind --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/ind_eng_eflomal_gdfa_all.conllu --test id_gsd-ud-test_2_5.conllu
"""
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
parser = argparse.ArgumentParser()
parser.add_argument("--bronze", default=None, type=str, required=True, help="Specify bronze number [1,2,3]")
parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
parser.add_argument("--train", default=None, type=str, required=True, help="train file")
parser.add_argument("--test", default=None, type=str, required=True, help="test file")
parser.add_argument("--epochs", default=30, type=int, required=False, help="Number of epochs, default 30")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

from flair.embeddings import StackedEmbeddings, TransformerWordEmbeddings, CharacterEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import AdamW


def train(bronze, lang, train, test, epochs):

    # 1. get the corpus
    columns = {1: 'text', 3: 'upos'}

    # this is the folder in which train, test and dev files reside
    data_folder = ''
    corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                # train_file =   '/mounts/work/silvia/POS/'+train,
                                train_file =   train,
                                # test_file  =   '/nfs/datx/UD/v2_5/'+test,
                                # dev_file   =   '/nfs/datx/UD/v2_5/'+test,        
                                test_file  =   test,
                                dev_file   =   test,                           
                                comment_symbol="#",
                                column_delimiter="\t"
                                )

    # 2. what label do we want to predict?
    label_type = 'upos'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    # 4. initialize embeddings
    # embedding_types = [
        # CharacterEmbeddings(),
        # TransformerWordEmbeddings('xlm-roberta-base', pooling_operation='first_last') 
    # ]

    # embeddings = StackedEmbeddings(embeddings=embedding_types)  
    embeddings = TransformerWordEmbeddings('xlm-roberta-base', pooling_operation='first_last', 
                                            allow_long_sentences=False, max_length=512, truncation=True) #default: fine_tune: bool = False,
    # embeddings = TransformerWordEmbeddings('xlm-roberta-large', pooling_operation='first_last') #default: fine_tune: bool = False,
    # embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased', pooling_operation='first_last', 
    #                                         allow_long_sentences=False, max_length=512, truncation=True) #default: fine_tune: bool = False,
    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=128,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            # use_crf=True)
                            use_crf=False)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/'+lang+'-upos-xlmr-final-'+bronze,
                learning_rate=0.0001,
                # learning_rate=0.001, 
                mini_batch_size=256,
                # mini_batch_size=1,
                mini_batch_chunk_size=32,
                # XLMR large:
                # mini_batch_size=256,
                # mini_batch_chunk_size=16,
                optimizer=AdamW,
                max_epochs=epochs,#6,             
                # max_epochs=30,             
                patience=epochs, #6, # so that it trains for all the 8 epochs, regardless the dev
                # patience = 30, # so that it trains for all the 8 epochs, regardless the dev
                checkpoint=True)

    # load the model to evaluate
    tagger: SequenceTagger = SequenceTagger.load('resources/taggers/'+lang+'-upos-xlmr-final-'+bronze+'/final-model.pt')
    # run evaluation procedure
    result = tagger.evaluate(corpus.test, mini_batch_size=128, out_path=f"predictions.txt", gold_label_type='upos', num_workers=32)
    print(result) # this is the result to report, the final one.

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--bronze", default=None, type=str, required=True, help="Specify bronze number [1,2,3]")
    # parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
    # parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
    # parser.add_argument("--train", default=None, type=str, required=True, help="train file")
    # parser.add_argument("--test", default=None, type=str, required=True, help="test file")
    # args = parser.parse_args()


    bronze = "bronze"+str(args.bronze)
    lang = args.lang
    train(bronze, lang, args.train, args.test, args.epochs)
    print('Model saved in: resources/taggers/'+lang+'-upos-xlmr-final-bronze'+bronze)


if __name__ == "__main__":
    main()


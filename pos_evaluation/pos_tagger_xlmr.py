"""
POS tagger with XLMR embeddings

$ python3 pos_tagger_xlmr.py --bronze 1 --lang tam --gpu 5 --train --test

"""
from flair.embeddings import StackedEmbeddings, TransformerWordEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import os
import argparse
from torch.optim import AdamW


def train(bronze, lang, train, test):

    # 1. get the corpus
    columns = {1: 'text', 3: 'upos'}

    # this is the folder in which train, test and dev files reside
    data_folder = ''
    corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                # train_file =   '/mounts/work/silvia/POS/'+train,
                                train_file =   train,
                                test_file  =   '/nfs/datx/UD/v2_5/'+test,
                                dev_file   =   '/nfs/datx/UD/v2_5/'+test,                                
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
    #     TransformerWordEmbeddings('xlm-roberta-base', pooling_operation='first_last') 
    # ]

    # embeddings = StackedEmbeddings(embeddings=embedding_types)
    embeddings = TransformerWordEmbeddings('xlm-roberta-base', pooling_operation='first_last') 

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=128,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/'+lang+'-upos-xlmr-final-bronze'+bronze,
                learning_rate=0.0001,
                # learning_rate=0.001,
                mini_batch_size=256,
                mini_batch_chunk_size=32,
                optimizer=AdamW,
                max_epochs=8,             
                patience = 8, # so that it trains for all the 8 epochs, regardless the dev
                checkpoint=True)
      
    # load the model to evaluate
    tagger: SequenceTagger = SequenceTagger.load('resources/taggers/'+lang+'-upos-xlmr-final-bronze'+bronze+'/final-model.pt')
    # run evaluation procedure
    result = tagger.evaluate(corpus.test, mini_batch_size=128, out_path=f"predictions.txt", gold_label_type='upos', num_workers=32)
    print(result) # this is the result to report, the final one.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", default=None, type=int, required=True, help="Specify bronze number [1,2,3]")
    parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
    parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
    parser.add_argument("--train", default=None, type=str, required=True, help="train file")
    parser.add_argument("--test", default=None, type=str, required=True, help="test file")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    bronze = "bronze"+str(args.bronze)
    lang = args.lang
    train(bronze, lang, args.train, args.test)
    print('Model saved in: resources/taggers/'+lang+'-upos-xlmr-final-bronze'+bronze)





if __name__ == "__main__":
    main()


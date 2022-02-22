POS EVALUATION

- Bronze training set evaluation via POS tagging
	- create_train_dev.py : create train and dev sets from Bronze data
	- pos_tagger.py : BiLSTM+CRF, Fasttext embeddings + Character embeddings
	- pos_tagger_bert.py : BiLSTM+CRF, BERT embeddings
- high_res_pos.py : use Flair POS tagger to tag high-resource languages

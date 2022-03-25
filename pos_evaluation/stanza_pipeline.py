"""
Use STANZA POS tagger to tag high-resource languages.

To run change:
- code
- out
- bible

"""

import stanza
import logging

# code = 'it'
# code = 'nl'
# code = 'es'
# code = 'da'
# code = 'cs'
# code = 'de'
# code = 'fr'
# code = 'sv'
# code = 'pl'
# code = 'en'
# code = 'pt'
# code = 'ar' 
code = 'hi'
# code = 'fa'
# code = 'ga' # irish
# code = 'ru' 
# code = 'zh' 
# code = 'hu' 
# code = 'ur' 
# code = 'el' 
# code = 'he' 
# code = 'cs' # check

stanza.download(code)
def tag_sentence(sentences):
    logging.getLogger().setLevel(logging.CRITICAL)
    # nlp = stanza.Pipeline(lang='fa', processors='tokenize,mwt,pos', tokenize_pretokenized=True)
    # nlp = stanza.Pipeline(lang=code, processors='tokenize,mwt,pos', tokenize_pretokenized=True, pos_batch_size=300)
    nlp = stanza.Pipeline(lang=code, processors='tokenize,pos', tokenize_pretokenized=True, pos_batch_size=300)
    # print(sentences)
    doc = nlp(sentences)
    list_tags = []
    for i, sent in enumerate(doc.sentences):
        print(f'====== Sentence {i+1} tokens =======')
        app = []
        for word in sent.words:
            # print(word.text, word.upos)
            app.append([word.text,word.upos])
        list_tags.append(app)
    return list_tags
    # print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')

# read Persian file and tag it
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/tam-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/fin-x-bible-helfi.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/prs-x-bible-goodnews.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/ita-x-bible-2009.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/nld-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/nld-x-bible-2007.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/spa-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/spa-x-bible-hablahoi-latina.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/dan-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/ces-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/deu-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/fra-x-bible-louissegond.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/swe-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/pol-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/eng-x-bible-mixed.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/por-x-bible-versaointernacional.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/arb-x-bible.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/hin-x-bible-newworld.conllu", "w+")
out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/hin-x-bible-bsi.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/gle-x-bible.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/deu-x-bible-bolsinger.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/rus-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/zho-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/hun-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/urd-x-bible-2007.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/ell-x-bible-newworld.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/heb-x-bible-helfi.conllu", "w+")
# out = open("/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/ces-x-bible-newworld.conllu", "w+")


# bible_file = "/nfs/datc/pbc/prs-x-bible-goodnews.txt"
# bible_file = "/nfs/datc/pbc/tam-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/ita-x-bible-2009.txt"
# bible_file = "/nfs/datc/pbc/nld-x-bible-2007.txt" #############
# bible_file = "/nfs/datc/pbc/nld-x-bible-newworld.txt" #############
# bible_file = "/nfs/datc/pbc/spa-x-bible-hablahoi-latina.txt"
# bible_file = "/nfs/datc/pbc/spa-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/dan-x-bible-newworld.txt" #############
# bible_file = "/nfs/datc/pbc/ces-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/deu-x-bible-newworld.txt" # double check
# bible_file = "/nfs/datc/pbc/fra-x-bible-louissegond.txt"
# bible_file = "/nfs/datc/pbc/swe-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/pol-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/por-x-bible-versaointernacional.txt"
# bible_file = "/nfs/datc/pbc/arb-x-bible.txt"
# bible_file = "/nfs/datc/pbc/hin-x-bible-newworld.txt"
bible_file = "/nfs/datc/pbc/hin-x-bible-bsi.txt"
# bible_file = "/nfs/datc/pbc/gle-x-bible.txt"
# bible_file = "/nfs/datc/pbc/deu-x-bible-bolsinger.txt"
# bible_file = "/nfs/datc/pbc/rus-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/zho-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/hun-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/urd-x-bible-2007.txt"
# bible_file = "/nfs/datc/pbc/ell-x-bible-newworld.txt"
# bible_file = "/nfs/datc/pbc/ces-x-bible-newworld.txt"

# bible_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/eng-x-bible-mixed.txt"
# bible_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/heb-x-bible-helfi.txt"




# bible_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/fin-x-bible-helfi.txt"
sentences = ""
count = 0
MAX_LEN = 0
with open(bible_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        l = line.strip().split("\t")
        if len(l)<2:
            continue
        if len(l[1])>MAX_LEN:
            MAX_LEN = len(l[1])
        sentences+=l[1]+"\n"
        count+=1
        # if count==3:
        #     break

print(MAX_LEN)
upos_tags = tag_sentence(sentences)
print(len(upos_tags))

sent_num = 0
with open(bible_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        l = line.strip().rstrip().split("\t")
        if len(l)<2:
            continue
        out.write(F"# sent_id = {l[0]}\n")
        out.write(F"# text = {l[1]}\n")
        # upos_tags = tag_sentence(l[1])
        # l[1] = l[1].replace("  ", " ")
        for i,w in enumerate(l[1].split(" ")):

            if not w.lower()==upos_tags[sent_num][i][0].lower():
                print("Error!")
                print(w)
                print(l[1])
                print(l[1].split(" "))
                print(upos_tags[sent_num][i][0].lower())
                break
            out.write(F"{i+1}\t{w}\t{w.lower()}\t{upos_tags[sent_num][i][1]}\t-\t-\t-\t-\t-\t-\n")

        sent_num+=1
        out.write("\n")
        # if sent_num==3:
        #     break

out.close()

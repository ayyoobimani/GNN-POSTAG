# %%
import gensim, logging, sys, codecs, os
sys.path.insert(0, '../')
from my_utils import utils

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class PBCSentences(object):
    def __init__(self, editions_file="/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/splits/helfi_lang_list.txt"
                     , pbc_directory='/nfs/datc/pbc/'):

        self.editions_file = editions_file

        self.files = []
        self.file_langs = {}
        self.pbc_directory = pbc_directory

        with open(editions_file) as inf:
            for line in inf:
                line = line.strip().split('\t')
                
                self.files.append(line[1])
                self.file_langs[line[1]] = line[0][:3]
 
    def __iter__(self):
        for fname in self.files:
            utils.LOG.info(f'reading {self.files.index(fname)} {fname}')
            if 'eng' in fname :
                directory = '/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/'
            elif fname in ['heb-x-bible-helfi', 'fin-x-bible-helfi', 'grc-x-bible-helfi']:
                directory = '/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi'
            else:
                directory = self.pbc_directory
            for line in codecs.open(os.path.join(directory, fname + ".txt"), encoding='utf8'):
                if line.startswith('#'):
                    continue
                
                line = line.strip().split('\t')
                if len(line) < 2:
                    continue
                verse_id, tokens = line[0], line[1].split()
                t_sentences = [[verse_id, f'{self.file_langs[fname]}:{x}'] for x in tokens]
                t_sentences.append([f'{self.file_langs[fname]}:{x}' for x in tokens])
                for item in t_sentences:
                    yield item

sentences = PBCSentences(editions_file='/mounts/Users/student/ayyoob/Dokumente/code/POS-TAGGING/editions_listsmall2.txt')
model = gensim.models.Word2Vec(sentences=sentences, min_count=1, workers=80, sg=1, epochs=10)
model.save("/mounts/work/ayyoob/models/w2v/word2vec_POS_small_final_langs_10e.model")
print('done')
exit(0)

# %%
from gensim.models import Word2Vec
import torch
model = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_helfi_langs_15e.model")

print(len(list(model.wv.key_to_index.items())))

print(model.wv.vectors.shape)

word_vectors = torch.from_numpy(model.wv.vectors, dtype=torch.float, requires_grad=True)

print(word_vectors.shape)
model.wv.most_similar('eng:cheese')
model.wv.similarity('eng:god', 'deu:god')


# %%
import torch 
input = torch.tensor([[-1,-2,-2], [-4, -5, -6], [-7, -8, 100]], dtype=torch.float)

input2 = torch.tensor([[7], [8]], dtype=torch.float)
out = torch.tensor([[1,2,3, 8], [4,5,6, 9]], dtype=torch.float)

class mm(torch.nn.Module):
    def __init__(self, input_embedding):
        super(mm, self).__init__()
        self.emb = torch.nn.Embedding.from_pretrained(input_embedding,  freeze=False)
        self.layer = torch.nn.Linear(4,4)

    def forward(self):
        a = torch.cat((self.emb(torch.tensor([0,2], dtype=torch.long)), input2), dim=1)
        return self.layer(a)

mmm = mm(input)
print(mmm.parameters)
optimizer = torch.optim.Adam(mmm.parameters(), lr=0.1)
#optimizer.add_param_group({'params': input})
loss_f = torch.nn.MSELoss()

#a = torch.narrow(input, 0, 0, 2)
print(input)
for i in range(100):
    
    optimizer.zero_grad()
    #print(input)
    #print(layer(a))

    loss = loss_f(out, mmm())
    loss.backward()
    optimizer.step()

print(input)
#print(mmm.emb(torch.LongTensor([0,1,2])))
print(mmm())
print("")




import os
import codecs

# version 2_10
ud_files_map_2_10 = {}
# ud_files_map_2_10["amh"]="UD_Amharic-ATT/am_att-ud-test.conllu"  #####
# ud_files_map_2_10["ind"]="UD_Indonesian-GSD/id_gsd-ud-test.conllu" #######
# ud_files_map_2_10["pes"]="UD_Persian-Seraji/fa_seraji-ud-test.conllu" ####
# ud_files_map_2_10["por"]="UD_Portuguese-Bosque/pt_bosque-ud-test.conllu" ###
# ud_files_map_2_10["tur"]="UD_Turkish-IMST/tr_imst-ud-test.conllu" #####        
# ud_files_map_2_10["glv"]="UD_Manx-Cadhan/gv_cadhan-ud-test.conllu" ###
# ud_files_map_2_10["mar"]="UD_Marathi-UFAL/mr_ufal-ud-test.conllu" ####
# ud_files_map_2_10["yor"]="UD_Yoruba-YTB/yo_ytb-ud-test.conllu" ####
ud_files_map_2_10["myv"]="UD_Erzya-JR/myv_jr-ud-test.conllu"

ud_directory = '/nfs/datx/UD/v2_10/ud-treebanks-v2.10'
ud_files = list(ud_files_map_2_10.values())
version = 'v2_10'


# version 2_5
# ud_directory = '/nfs/datx/UD/v2_5/'
# ud_files = ['myv_jr-ud-test_2_5.conllu', 'mr_ufal-ud-test_2_5.conllu', 'gv_cadhan-ud-test_2_7.conllu', 'am_att-ud-test_2_5.conllu', 
#             'pt_bosque-ud-test_2_5.conllu', 'fa_seraji-ud-test_2_5.conllu', 'tr_imst-ud-test_2_5.conllu', 'id_gsd-ud-test_2_5.conllu'] #Erzya, marathi, manx
# version = 'v2_5'

# # version 2_3
# ud_directory = "/nfs/datx/UD/v2_3/"
# ud_files = [ "am_att-ud-test_2_3.conllu", "pt_pud-ud-test_2_3.conllu", "myv_jr-ud-test_2_3.conllu", "tr_pud-ud-test_2_3.conllu", "yo_ytb-ud-test_2_3.conllu", 
#             'id_pud-ud-test_2_3.conllu']
# version = 'v2_3'


# # version 2
# ud_directory = "/nfs/datx/UD/v2_0/"
# ud_files = ["pt-ud-test_2_0.conllu"]
# version = 'v2_0'

# # version 1_2
# ud_directory = "/nfs/datx/UD/v1_2/universal-dependencies-1.2/"
# ud_files = ["UD_Persian/fa-ud-test.conllu", "UD_Portuguese/pt-ud-test.conllu"]
# version = 'v1_2'

out_directory = '/mounts/data/proj/ayyoob/POS_tagging/dataset/UD'


if not os.path.exists(f'{out_directory}/{version}/'):
    os.makedirs(f'{out_directory}/{version}/')

class Node:
    def __init__(self, number, parent_number, is_dependent, text, pos_tag, dependency_groups, actual_words):
        self.number = number
        self.parent_number = parent_number
        self.is_dependent = is_dependent
        self.text = text
        self.pos_tag = pos_tag
        self.parent = None
        
        if self.is_dependent:
            self.dependency_group = dependency_groups[-1]
            self.text = actual_words[-1]

        self.distance_to_head = -1

    def _get_distance_to_head(self):
        res = 0
        tmp_n = self
        while tmp_n.parent != None:
            tmp_n = tmp_n.parent
            res += 1

        return res

    def get_distance_to_head(self):
        if self.distance_to_head == -1:
            self.distance_to_head = self._get_distance_to_head()
        
        return self.distance_to_head

def create_tree(all_nodes):
    for node in all_nodes:
        if node.parent_number != 0:
            node.parent = all_nodes[node.parent_number-1]

def is_head_node(node, all_nodes):
    min_nom = 999999999
    head = None
    for node_nom in node.dependency_group:
        tmp_n = all_nodes[node_nom-1]
        if tmp_n.get_distance_to_head() < min_nom:
            min_nom = tmp_n.get_distance_to_head()
            head = tmp_n
    
    if head == node:
        return True
    return False

def process_sentence(all_nodes, out_file, sent_id_line=None, text_line=None):
    if sent_id_line != None:
        fo.write(f'{sent_id_line}\n')
    if text_line != None:
        fo.write(f'{text_line}\n')

    create_tree(all_nodes)

    for node in all_nodes:
        if (not node.is_dependent) or is_head_node(node, all_nodes):
            fo.write(f'{node.number}\t{node.text}\t{node.text}\t{node.pos_tag}\n')

    fo.write('\n')


def process_line(all_nodes, line, multi_word_groups, actual_words):
    splits = line.split()
    if '-' in splits[0]:
        beginning = int(splits[0].split('-')[0])
        end = int(splits[0].split('-')[1])
        multi_word_groups.append(list(range(beginning, end+1)))
        actual_words.append(splits[1])
        return
    
    node_nom = int(splits[0])
    is_dependent = False
    if len(multi_word_groups) > 0 and node_nom in multi_word_groups[-1]:
        is_dependent = True
    parent_number = int(splits[6])

    node = Node(node_nom, parent_number,is_dependent, splits[1], splits[3], multi_word_groups, actual_words)
    all_nodes.append(node)

for ud_file in ud_files:
    ud_path = f'{ud_directory}/{ud_file}'
    with codecs.open(ud_path, 'r', 'utf-8') as fi, codecs.open(f'{out_directory}/{version}/{ud_file.split("/")[-1]}', 'w', 'utf-8') as fo:

        all_nodes = []
        multi_word_groups = []
        actual_words = []
        sent_id_line = None
        text_line = None
        for line in fi:
            line = line.strip()
            
            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    sent_id_line = line
                if line.startswith('# text'):
                    text_line = line
                continue
            if line == '':
                process_sentence(all_nodes, fo, sent_id_line, text_line)
                all_nodes = []
                multi_word_groups = []
                actual_words = []
                continue
            
            process_line(all_nodes, line, multi_word_groups, actual_words)
            

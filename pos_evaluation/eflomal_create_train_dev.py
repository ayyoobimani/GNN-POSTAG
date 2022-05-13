"""
Adapted code from Masoud
/mounts/work/mjalili/projects/flair_pos_tagger/pbc_corpus_maker.py

Transfer from English:
$ python3 eflomal_create_train_dev.py --lang hin --bible hin-x-bible-bsi.txt --out /mounts/work/silvia/POS/eflomal/prova --transfer 1

Transfer from all:
python3 eflomal_create_train_dev.py --lang hin --bible hin-x-bible-bsi.txt --out /mounts/work/silvia/POS/eflomal/prova --transfer 0

"""

import os
import codecs
import argparse
import collections
from platform import python_branch
from tkinter.tix import Tree
from tqdm import tqdm


def align_files(out_folder, src_lang="", trg_lang="", aligner="itermax", device='cpu'):
	print("Aligning...")
	print(aligner)

	if aligner in ["inter", "itermax"]:
		aligner_path = "python /mounts/Users/student/masoud/projects/simalign/scripts/align_files.py"
	elif aligner in ["fast", "eflomal"]:
		aligner_path = "python3 /mounts/work/silvia/pbc_utils/extract_alignments.py"

	# src_path = F"/mounts/work/silvia/POS/eflomal/src_tmp.txt"
	# trg_path = F"/mounts/work/silvia/POS/eflomal/trg_tmp.txt"	
	src_path = out_folder+"/src_tmp.txt"
	trg_path = out_folder+"/trg_tmp.txt"

	# out_path = F"data/aligns/pbc_yo/{src_lang}_{trg_lang}_align"
	# out_path = F"/mounts/work/silvia/POS/eflomal/ALIGNMENTS/{src_lang}_{trg_lang}"
	out_path = F"{out_folder}/ALIGNMENTS/{src_lang}_{trg_lang}"

	if aligner in ["inter", "itermax"]:
		os.system(aligner_path + F" {src_path} {trg_path} --matching-methods=fair --add-probs -device={device} -output={out_path}.simalign")
	elif aligner in ["fast", "eflomal"]:
		print(aligner_path + F" -s {src_path} -t {trg_path} -m {aligner} -o {out_path}.{aligner}")
		os.system(aligner_path + F" -s {src_path} -t {trg_path} -m {aligner} -o {out_path}.{aligner}")


def create_pos_with_aligns(out_folder, lang_list, prfs, cover, all_trg_sents, all_verse_map, target_lang, flag, aligner="itermax"):
	print("creating conllu files with alignments...")
	print(aligner)
	all_tags = {}
	all_aligns = {}

	for lang, prf in zip(lang_list, prfs):
		# align_path = F"/mounts/work/silvia/POS/eflomal/ALIGNMENTS/{prf}_{target_lang}"
		align_path = F"{out_folder}/ALIGNMENTS/{prf}_{target_lang}"

		if aligner in ["inter", "itermax", "rev", "mwmf"]:
			align_path += ".simalign." + aligner
			# align_path += "." + aligner
		elif "rev" in aligner or "fwd" in aligner or "gnn" in aligner:
			align_path += F".{aligner}"
		else:
			align_path += F".{aligner}"
		
		print(align_path)
		with open(align_path, "r") as f_al:
			aligns = [l.strip() for l in f_al.readlines()]
			if "gnn" in align_path:
				aligns = {l.split("\t")[0]: [p.split("-") for p in l.split("\t")[1].split()] if "\t" in l else [] for l in aligns}

			elif "simalign" in align_path or "eflomal" in align_path:
				aligns = [[p.split("-") for p in l.split("\t")[1].split()] if "\t" in l else [] for l in aligns]

			else:
				aligns = [[p.split("-") for p in l.split()] for l in aligns]

			# check for probs
			if "gnn" in align_path:
				aligns = {v: [((int(p[0]), int(p[1])), [1.0]) for p in aligns[v]] for v in aligns}
			elif len(aligns[0][0]) > 2:
				aligns = [[((int(p[0]), int(p[1])), [float(x) for x in p[2:]]) for p in l] for l in aligns]
			else:
				aligns = [[((int(p[0]), int(p[1])), [1.0]) for p in l] for l in aligns]

			# map to verses
			if "gnn" in align_path:
				all_aligns[prf] = aligns
			else:
				all_aligns[prf] = {all_verse_map[prf][i][0]: l for i, l in enumerate(aligns)}

		
		# Use POS tagged high-resource files
		all_tags[prf] = {}
		# with codecs.open(F"data/pbc_yo/raw_tagged/{lang[:3]}.conllu", "r", "utf-8") as lang_pos:
		# with codecs.open(F"data/pbc_yor_10/all_pos_tags/{lang}.conllu", "r", "utf-8") as lang_pos:
		# with codecs.open(F"/mounts/work/mjalili/projects/flair_pos_tagger/data/pbc_yor_10/all_pos_tags/{lang}.conllu", "r", "utf-8") as lang_pos:
		with codecs.open(F"/mounts/work/silvia/POS/TAGGED_LANGS/STANZA/{lang}.conllu", "r", "utf-8") as lang_pos:
			# print(F"{lang}.conllu")
			tag_sent = []
			sent_id = ""
			for sline in lang_pos:
				sline = sline.strip()
				if sline == "":
					if sent_id not in all_aligns[prf]:
						tag_sent = []
						sent_id = ""
						continue

					all_tags[prf][sent_id] = [p[3] for p in tag_sent]
					tag_sent = []
					sent_id = ""
				# elif "# verse_id" in sline:
				elif "# sent_id" in sline:
					sent_id = sline.split()[-1]
				elif sline[0] == "#":
					continue
				else:
					tag_sent.append(sline.split("\t"))

	print("writing...")

	# if flag: # =1 = transfer only from english
	# 	f_trg_pos_train = codecs.open(F"/mounts/work/silvia/POS/eflomal/{target_lang}_eng_{aligner.replace('.', '_')}_train.conllu", "w", "utf-8")
	# 	f_trg_pos_dev = codecs.open(F"/mounts/work/silvia/POS/eflomal/{target_lang}_eng_{aligner.replace('.', '_')}_dev.conllu", "w", "utf-8")
	# else:
	# 	f_trg_pos_train = codecs.open(F"/mounts/work/silvia/POS/eflomal/{target_lang}_all_{aligner.replace('.', '_')}_train.conllu", "w", "utf-8")
	# 	f_trg_pos_dev = codecs.open(F"/mounts/work/silvia/POS/eflomal/{target_lang}_all_{aligner.replace('.', '_')}_dev.conllu", "w", "utf-8")
	if flag: # =1 = transfer only from english
		f_trg_pos_train = codecs.open(F"{out_folder}/{target_lang}_eng_{aligner.replace('.', '_')}_train.conllu", "w", "utf-8")
		f_trg_pos_dev = codecs.open(F"{out_folder}/{target_lang}_eng_{aligner.replace('.', '_')}_dev.conllu", "w", "utf-8")
	else:
		f_trg_pos_train = codecs.open(F"{out_folder}/{target_lang}_all_{aligner.replace('.', '_')}_train.conllu", "w", "utf-8")
		f_trg_pos_dev = codecs.open(F"{out_folder}/{target_lang}_all_{aligner.replace('.', '_')}_dev.conllu", "w", "utf-8")

	dev_portion = int(len(all_trg_sents)*10/100) # TODO: add shuffle somewhere	? 
                                                 # TODO: merge train and dev for training ?
	sent_id = 0
	train_size, dev_size = 0, 0

	for verse in all_trg_sents:
		if len(cover[verse]) < 1:
			continue
		trg_sent = all_trg_sents[verse].split()
		trg_tags = [collections.defaultdict(float, {"X": 0.0}) for w in trg_sent]

		for lang, sid in cover[verse]:
			# print(lang)
			if verse not in all_aligns[lang]: continue
			try:
				for al_pair in all_aligns[lang][verse]:
					
					# if len(al_pair)<2:
					# 	continue
					trg_tags[al_pair[0][1]][all_tags[lang][verse][al_pair[0][0]]] += al_pair[1][0]
			except:
				print(al_pair[0][1])
				print(al_pair[1][0])
				print("al_pair[0][0] ",al_pair[0][0])
				# print(all_tags[lang])
				print(all_tags[lang][verse])
				# print(all_tags[lang][verse][al_pair[0][0]]) # error
				# print(trg_tags[al_pair[0][1]][all_tags[lang][verse][al_pair[0][0]]])

		trg_tags = [max(trg_tags[i], key=lambda x: (trg_tags[i][x], x) if x != "X" else (-1.0, "X")) for i in range(len(trg_tags))]

		# if sum([1 if tag == "X" else 0 for tag in trg_tags]) > 0.5 * len(trg_tags):
		# 	continue

		if sent_id < dev_portion:
			dev_size += 1
			f_trg_pos_dev.write(F"# verse_id = {verse}\n")
			f_trg_pos_dev.write(F"# sent_id = {sent_id}\n")
			f_trg_pos_dev.write(F"# text = " + " ".join(trg_sent) + "\n")
			for i, w in enumerate(trg_sent):
				line_str = F"{i+1}\t{w}\t{w}\t{trg_tags[i]}\t_\t_\t0\t_\t_\t_\n"
				f_trg_pos_dev.write(line_str)
			f_trg_pos_dev.write("\n")
		else:
			train_size += 1
			f_trg_pos_train.write(F"# verse_id = {verse}\n")
			f_trg_pos_train.write(F"# sent_id = {sent_id}\n")
			f_trg_pos_train.write(F"# text = " + " ".join(trg_sent) + "\n")
			for i, w in enumerate(trg_sent):
				line_str = F"{i+1}\t{w}\t{w}\t{trg_tags[i]}\t_\t_\t0\t_\t_\t_\n"
				f_trg_pos_train.write(line_str)
			f_trg_pos_train.write("\n")
		sent_id += 1

	f_trg_pos_train.close()
	f_trg_pos_dev.close()
	print("Train size: ", train_size)
	print("Dev size: ", dev_size)


def create_pos_with_tag_file(all_trg_sents, tag_file=""):
	import torch

	print("creating conllu files with alignments...")
	postag_map = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, 
				"NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, 
				"SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
	postags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", 
				"PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

	if tag_file:
		tag_img = torch.load(tag_file, "cpu")
	else:
		# tag_img = torch.load("/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags.pickle", "cpu")
		tag_img = torch.load("/mounts/work/ayyoob/results/gnn_align/yoruba/pos_tags_tagFeatureVectors_TrainOverConfidentNodes.pickle", "cpu")

	# import ipdb; ipdb.set_trace()
	tag_verses = [v for v in tag_img.keys()]

	print("writing...")

	f_trg_pos_train = codecs.open(F"data/pbc_yor_10/yor_out_tags/yor_10_12_train.conllu", "w", "utf-8")
	f_trg_pos_dev = codecs.open(F"data/pbc_yor_10/yor_out_tags/yor_10_12_dev.conllu", "w", "utf-8")

	sent_id = 0
	for verse in all_trg_sents:
		if verse not in tag_img:
			continue
		trg_sent = all_trg_sents[verse].split()
		trg_tags = ["X" for w in trg_sent]

		for wid in tag_img[verse]:
			wpos = postags[tag_img[verse][wid]]
			trg_tags[wid] = wpos

		# import ipdb; ipdb.set_trace()

		if sum([1 if tag == "X" else 0 for tag in trg_tags]) > 0.5 * len(trg_tags):
			continue

		if sent_id < 3000:
			f_trg_pos_dev.write(F"# verse_id = {verse}\n")
			f_trg_pos_dev.write(F"# sent_id = {sent_id}\n")
			f_trg_pos_dev.write(F"# text = " + " ".join(trg_sent) + "\n")
			for i, w in enumerate(trg_sent):
				line_str = F"{i+1}\t{w}\t{w}\t{trg_tags[i]}\t_\t_\t0\t_\t_\t_\n"
				f_trg_pos_dev.write(line_str)
			f_trg_pos_dev.write("\n")
		else:
			f_trg_pos_train.write(F"# verse_id = {verse}\n")
			f_trg_pos_train.write(F"# sent_id = {sent_id}\n")
			f_trg_pos_train.write(F"# text = " + " ".join(trg_sent) + "\n")
			for i, w in enumerate(trg_sent):
				line_str = F"{i+1}\t{w}\t{w}\t{trg_tags[i]}\t_\t_\t0\t_\t_\t_\n"
				f_trg_pos_train.write(line_str)
			f_trg_pos_train.write("\n")
		sent_id += 1

	f_trg_pos_train.close()
	f_trg_pos_dev.close()



# --------------------------------------------------------
#		MAIN
# --------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Make dataset from pbc", epilog="")
	# parser.add_argument("--tag_type", type=str, default="upos")
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--aligner", type=str, default="eflomal") #inter, eflomal
	parser.add_argument("--lang", type=str, required=True)
	parser.add_argument("--bible", type=str, required=True)
	parser.add_argument("--out", type=str, required=True, help="Output folder")
	parser.add_argument("--transfer", type=int, required=True, help="[1] English, [0] All")

	args = parser.parse_args()
	if args.transfer == 0:
		# lang_list = ["eng-x-bible-newworld2013", "deu-x-bible-newworld", "ces-x-bible-newworld",
		# 			"fra-x-bible-newworld", "hin-x-bible-newworld", "ita-x-bible-2009",
		# 			# "prs-x-bible-goodnews", 
		# 			"ron-x-bible-2006", "spa-x-bible-newworld"]

		lang_list = [
			"eng-x-bible-mixed",
			"deu-x-bible-bolsinger",
			"rus-x-bible-newworld",
			"dan-x-bible-newworld",
			"fin-x-bible-helfi",
			"gle-x-bible",
			"pol-x-bible-newworld",
			"swe-x-bible-newworld",
			"ita-x-bible-2009",
			"fra-x-bible-louissegond",
			"spa-x-bible-hablahoi-latina",
			"zho-x-bible-newworld",
			"arb-x-bible",
			"tam-x-bible-newworld",
			"urd-x-bible-2007"
		]
		flag = 0
	else:
		lang_list = ["eng-x-bible-mixed"]
		# lang_list = ["eng-x-bible-newworld2013"]
		flag = 1

	lang_prfs = [l for l in lang_list]

	# save target language bible
	bible_sents = {}
	id_cover = collections.defaultdict(lambda: [])

	if args.bible in ['fin-x-bible-helfi.txt', 'heb-x-bible-helfi.txt', 'eng-x-bible-mixed.txt']:
		bible_path = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/"+args.bible
	else:
		bible_path = "/nfs/datc/pbc/"+args.bible

    # Save bible
	with codecs.open(bible_path, "r", "utf-8") as trg_file:
		for l in trg_file:
			if l[0] == "#":
				continue
			l = l.strip().split("\t")
			if len(l) != 2:
				continue
			bible_sents[l[0]] = l[1]

    # Save sentences of the source languages
	all_sent_pairs = {}
	for lang_id, lang in enumerate(lang_list):
		prefix = lang_prfs[lang_id]
		print(F"{prefix}-{args.lang}...")
		all_sent_pairs[prefix] = []
        
		if lang in ['fin-x-bible-helfi', 'heb-x-bible-helfi', 'eng-x-bible-mixed']:
		    lang_path = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/"+lang+".txt"
		else:
		    lang_path = "/nfs/datc/pbc/"+lang+".txt"

		# with codecs.open(F"/mounts/work/mjalili/projects/flair_pos_tagger/data/pbc_yor_10/text/{lang}.txt", "r", "utf-8") as src_file:
		with codecs.open(lang_path, "r", "utf-8") as src_file:
			for l in src_file:
				if l[0] == "#":
					continue
				l = l.strip().split("\t")
				if len(l) != 2:
					continue
				if l[0] in bible_sents: # if id in my bible file
					all_sent_pairs[prefix].append([l[0], l[1]])
					id_cover[l[0]].append((prefix, len(all_sent_pairs[prefix]) - 1))

		#  Make alignments
		# with codecs.open("eflomal/src_tmp.txt", "w", "utf-8") as srcf, codecs.open("eflomal/trg_tmp.txt", "w", "utf-8") as trgf:
		with codecs.open(args.out+"/src_tmp.txt", "w", "utf-8") as srcf, codecs.open(args.out+"/trg_tmp.txt", "w", "utf-8") as trgf:
			for verse in all_sent_pairs[prefix]:
				srcf.write("{}\t{}\n".format(verse[0], verse[1]))
				trgf.write("{}\t{}\n".format(verse[0], bible_sents[verse[0]]))

		align_files(args.out, prefix, args.lang, aligner=args.aligner, device=args.device)
	
	print("\nAlignments done.\n")
	# create_pos_with_aligns(lang_prfs, id_cover, bible_sents, all_sent_pairs, args.lang, aligner=args.aligner+".gdfa")
	create_pos_with_aligns(args.out, lang_list, lang_prfs, id_cover, bible_sents, all_sent_pairs, args.lang, flag, aligner=args.aligner+".gdfa")
	# create_pos_with_tag_file(bible_sents)


        

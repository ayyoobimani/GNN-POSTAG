import os, sys
import os.path
import regex
import codecs
import collections
from random import shuffle
import concurrent.futures
import logging
import time
import argparse
from app import utils, general_align_reader





def setup_dict_entry(_dict, entry, val):
	if entry not in _dict:
		_dict[entry] = val

def read_alignment_file(file_path):
	res = []
	with open(file_path, 'r') as f:
		for line in f:
			s_l = line.split('\t') # handle index
			if len(s_l) > 1:
				res.append(s_l[1])
			else:
				res.append(s_l[0])
	
	return res

def add_tokens(tokens, token_dict):
	for t in tokens:
		setup_dict_entry(token_dict, t, 0)
		token_dict[t] += 1

def add_edition_tokens(tokens, edition_dict, edition):
	setup_dict_entry(edition_dict, edition, {})
	add_tokens(tokens, edition_dict[edition])

def write_dict_data_to_file(file_path, data, mode):
	with open(file_path, mode) as of:
			for key in data:
				of.write(str(key) + "\t" + str(data[key]) + "\n")
				
def log_state(src_lang, trg_lang, state):
	logging.info(F"alignment process of {src_lang},{trg_lang} {state}")

def get_in_order(lang_name1, lang_name2, files1, files2, store_lang1_stat, store_lang2_stat):
	s_lang, t_lang = align_reader.get_ordered_langs(lang_name1, lang_name2)
	if s_lang == lang_name1:
		return (lang_name1, lang_name2, files1, files2, store_lang1_stat, store_lang2_stat)
	elif t_lang == lang_name1:
		return (lang_name2, lang_name1, files2, files1, store_lang2_stat, store_lang1_stat)

def compute_alignment_statics(lang_name1, lang_name2, files1, files2, store_lang1_stat, store_lang2_stat):
	src_lang_name, trg_lang_name, src_files, trg_files, store_slang_stat, store_tlang_stat = get_in_order(lang_name1, lang_name2, files1, files2, store_lang1_stat, store_lang2_stat)
	
	log_state(src_lang_name, trg_lang_name, "starting, sfiles:%d, tfiles: %d" % (len(src_files), len(trg_files)) )

	if os.path.exists("{}/{}_{}_tokens_stat.txt".format(lang_pair_stats_dir,src_lang_name, trg_lang_name)):
		log_state(src_lang_name, trg_lang_name, "early abort")
		return
	src_sentences = utils.read_files(src_files)
	trg_sentences = utils.read_files(trg_files)

	src_lang_tokens = {}
	target_lang_tokens = {}
	src_edition_tokens = {}
	target_edition_tokens = {}

	lang_pair_freqs = {}
	edition_pair_freqs = {}
	lang_pair_verse_count = 0
	edition_pair_verse_count = {}

	alignments = read_alignment_file("{}/{}_{}_word.inter".format(alignment_path, src_lang_name, trg_lang_name))

	#----------------------------- calculating stats -------------------------------------------#
	log_state(src_lang_name, trg_lang_name, "calculating stats")

	for sfile in src_sentences:
			for verse in src_sentences[sfile]:
				s_terms = src_sentences[sfile][verse].split()
				add_tokens(s_terms, src_lang_tokens)
				add_edition_tokens(s_terms, src_edition_tokens, sfile)
				for tfile in trg_sentences:
					if verse in trg_sentences[tfile]:
						t_terms = trg_sentences[tfile][verse].split()
						aling_pairs = alignments[lang_pair_verse_count].split()
						
						edition_pair = sfile + "_" + tfile
						setup_dict_entry(edition_pair_verse_count, edition_pair, 0)
						edition_pair_verse_count[edition_pair] += 1
						setup_dict_entry(edition_pair_freqs, edition_pair, {})
						

						for align_pair in aling_pairs:
							indices = [ int(x) for x in align_pair.split('-') ]
							term_pair = s_terms[indices[0]] + '\t' + t_terms[indices[1]]

							setup_dict_entry(lang_pair_freqs, term_pair, 0)
							lang_pair_freqs[term_pair] += 1
							setup_dict_entry(edition_pair_freqs[edition_pair], term_pair, 0)
							edition_pair_freqs[edition_pair][term_pair] += 1

						lang_pair_verse_count += 1

	if store_tlang_stat:
		for tfile in trg_sentences:
			for verse in trg_sentences[tfile]:
				# print(tfile, verse)
				# print(trg_sentences[tfile][verse])
				t_terms = trg_sentences[tfile][verse].split()
				add_tokens(t_terms, target_lang_tokens)
				add_edition_tokens(t_terms, target_edition_tokens, tfile)

	#-------------------------------- writing stats -----------------------------------------#
	log_state(src_lang_name, trg_lang_name, "writing stats")
	# try:
	if store_slang_stat:
		write_dict_data_to_file(language_token_stats_file, {src_lang_name: len(src_lang_tokens)}, 'a')
		write_dict_data_to_file("{}/{}_tokens_stat.txt".format(lang_stats_dir, src_lang_name), src_lang_tokens, 'w')

		write_dict_data_to_file(edition_token_stats_file, {x: len(src_edition_tokens[x]) for x in src_edition_tokens}, 'a')
		for edition in src_edition_tokens:
			write_dict_data_to_file("{}/{}_tokens_stat.txt".format(edition_stats_dir, edition), src_edition_tokens[edition], 'w')
		
		write_dict_data_to_file(lang_verse_stat_file, {src_lang_name: sum([len(src_sentences[x]) for x in src_sentences])}, 'a')
		write_dict_data_to_file(edition_verse_stat_file, {x: len(src_sentences[x]) for x in src_sentences}, 'a')
	
	if store_tlang_stat:
		write_dict_data_to_file(language_token_stats_file, {trg_lang_name: len(target_lang_tokens)}, 'a')
		write_dict_data_to_file("{}/{}_tokens_stat.txt".format(lang_stats_dir, trg_lang_name), target_lang_tokens, 'w')

		write_dict_data_to_file(edition_token_stats_file, {x: len(target_edition_tokens[x]) for x in target_edition_tokens}, 'a')
		for edition in target_edition_tokens:
			write_dict_data_to_file("{}/{}_tokens_stat.txt".format(edition_stats_dir, edition), target_edition_tokens[edition], 'w')
		
		write_dict_data_to_file(lang_verse_stat_file, {trg_lang_name: sum([len(trg_sentences[x]) for x in trg_sentences])}, 'a')
		write_dict_data_to_file(edition_verse_stat_file, {x: len(trg_sentences[x]) for x in trg_sentences}, 'a')
	
	write_dict_data_to_file(lang_pair_token_count_stats_file, {"{}_{}".format(src_lang_name, trg_lang_name): len(lang_pair_freqs)}, 'a')
	write_dict_data_to_file(lang_pair_token_totcount_stats_file, {"{}_{}".format(src_lang_name, trg_lang_name): sum(lang_pair_freqs.values())}, 'a')
	write_dict_data_to_file("{}/{}_{}_tokens_stat.txt".format(lang_pair_stats_dir,src_lang_name, trg_lang_name), lang_pair_freqs, 'w')

	for edition_pair in edition_pair_freqs:
		write_dict_data_to_file(edition_pair_token_count_stats_file, {edition_pair:len(edition_pair_freqs[edition_pair])}, 'a')
		write_dict_data_to_file(edition_pair_token_totcount_stats_file, {edition_pair:sum(edition_pair_freqs[edition_pair].values())}, 'a')
		write_dict_data_to_file("{}/{}_tokens_stat.txt".format(edition_pair_stats_dir,edition_pair), edition_pair_freqs[edition_pair], 'w')

	write_dict_data_to_file(lang_pair_verse_stat_file, {"{}_{}".format(src_lang_name, trg_lang_name): lang_pair_verse_count}, 'a')
	write_dict_data_to_file(edition_pair_verse_stat_file, edition_pair_verse_count, 'a')
	# except Exception as exc:
	# 	logging.error("error while writing stats : %s", exc)
	log_state(src_lang_name, trg_lang_name, "end")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="compute alignment statistics for languages mentioned in lang_files.txt file.", 
	epilog="example: python eflomal_align_maker.py -s 0 -e 200 \n python eflomal_align_maker.py -l eng")
	parser.add_argument("-s", default=0, help="Start counter for number of files to process")
	parser.add_argument("-e", default=sys.maxsize, help="End counter for number of files to process")
	parser.add_argument("-l", default="", help="List of langs to compute stats for. If not provided, all langs are considered")
	parser.add_argument("-w", default=1, help="Number of cpu workers to use for stat calculation")


	
	args = parser.parse_args()
	utils.setup(os.environ['CONFIG_PATH'])
	
	format = "%(asctime)s: %(message)s"
	logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

	align_reader = general_align_reader.GeneralAlignReader()
	lang_files = align_reader.lang_files
	all_langs = align_reader.all_langs

	logging.info("language count: %d", len(all_langs))
	logging.info("files count: %d", sum([len(x) for x in lang_files.values()]))


	#######################    setting paths    #############################
	alignment_path = utils.alignments_dir

	language_token_stats_file = f"{utils.stats_directory}/lang_token_stats.txt"
	edition_token_stats_file = f"{utils.stats_directory}/edition_token_stats.txt"
	lang_verse_stat_file = f"{utils.stats_directory}/lang_verse_stats.txt"
	edition_verse_stat_file = f"{utils.stats_directory}/edition_verse_stats.txt"
	lang_pair_token_count_stats_file = f"{utils.stats_directory}/lang_pair_token_count_stats.txt"
	lang_pair_token_totcount_stats_file = f"{utils.stats_directory}/lang_pair_token_totcount_stats.txt"
	edition_pair_token_count_stats_file = f"{utils.stats_directory}/edition_pair_token_count_stats.txt"
	edition_pair_token_totcount_stats_file = f"{utils.stats_directory}/edition_pair_token_totcount_stats.txt"
	lang_pair_verse_stat_file = f"{utils.stats_directory}/lang_pair_verse_stats.txt"
	edition_pair_verse_stat_file = f"{utils.stats_directory}/edition_pair_verse_stats.txt"

	lang_pair_stats_dir = f"{utils.stats_directory}/lang_pair_stats"
	edition_pair_stats_dir = f"{utils.stats_directory}/edition_pair_stats"
	lang_stats_dir = f"{utils.stats_directory}/lang_stats"
	edition_stats_dir = f"{utils.stats_directory}/edition_stats"

	#################### start actual stat calc ##############################
	a_slang = []
	a_tlang = []
	a_sfiles = []
	a_tfiles = []
	s_lang_store = []
	t_lang_store = []
	if args.l != "":
		if args.l not in all_langs:
			logging.error("bad language selected")
		else:
			for i, t_lang in enumerate(all_langs):
				if t_lang != args.l:
					a_slang.append(args.l)
					a_tlang.append(t_lang)
					a_sfiles.append(lang_files[args.l])
					a_tfiles.append(lang_files[t_lang])
					t_lang_store.append(True)
					if i == 0:
						s_lang_store.append(True)
						compute_alignment_statics(args.l, t_lang, lang_files[args.l], lang_files[t_lang], True, True)
					else:
						s_lang_store.append(False)
						compute_alignment_statics(args.l, t_lang, lang_files[args.l], lang_files[t_lang], False, True)

	else:
		for i, s_lang in enumerate(all_langs):
			if i < int(args.s) or i>int(args.e):
				continue
			for j, t_lang in enumerate(all_langs[i+1:]):
				a_slang.append(s_lang)
				a_tlang.append(t_lang)
				a_sfiles.append(lang_files[s_lang])
				a_tfiles.append(lang_files[t_lang])

				t_lang_store.append(False)
				if j == 0:
					s_lang_store.append(True)
				else:
					s_lang_store.append(False)


		with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.w)) as executor:
			for r in executor.map(compute_alignment_statics, a_slang, a_tlang, a_sfiles, a_tfiles, s_lang_store, t_lang_store):
				try:
					print(r)
				except Exception as exc:
					print('generated an exception: %s' % (exc))




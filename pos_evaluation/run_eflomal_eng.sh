#! /bin/bash
echo "Hello World"
export CUDA_VISIBLE_DEVICES=0,1,2

python3 pos_tagger_xlmr.py --bronze eflomal --lang hin --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/hin_eng_eflomal_gdfa_all.conllu --test hi_hdtb-ud-test_2_5.conllu > out_hin.txt &
python3 pos_tagger_xlmr.py --bronze eflomal --lang por --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/por_eng_eflomal_gdfa_all.conllu --test pt_bosque-ud-test_2_5.conllu > out_por.txt &
python3 pos_tagger_xlmr.py --bronze eflomal --lang ind --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/ind_eng_eflomal_gdfa_all.conllu --test id_gsd-ud-test_2_5.conllu > out_ind.txt &
wait
python3 pos_tagger_xlmr.py --bronze eflomal --lang tur --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/tur_eng_eflomal_gdfa_all.conllu --test tr_imst-ud-test_2_5.conllu > out_tur.txt &
#python3 pos_tagger_xlmr.py --bronze eflomal --lang pes --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/pes_eng_eflomal_gdfa_all.conllu --test fa_seraji-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang afr --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/afr_eng_eflomal_gdfa_all.conllu --test af_afribooms-ud-test_2_5.conllu												
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal --lang amh --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/amh_eng_eflomal_gdfa_all.conllu --test am_att-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang eus --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/eus_eng_eflomal_gdfa_all.conllu --test eu_bdt-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang bul --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/bul_eng_eflomal_gdfa_all.conllu --test bg_btb-ud-test_2_5.conllu												
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal --lang lit --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/lit_eng_eflomal_gdfa_all.conllu --test lt_alksnis-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang tel --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/tel_eng_eflomal_gdfa_all.conllu --test te_mtg-ud-test_2_5.conllu 												TODO: reduce batch
#python3 pos_tagger_xlmr.py --bronze eflomal --lang bam --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/bam_eng_eflomal_gdfa_all.conllu --test bm_crb-ud-test_2_5.conllu 												
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal --lang bel --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/bel_eng_eflomal_gdfa_all.conllu --test be_hse-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang myv --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/myv_eng_eflomal_gdfa_all.conllu --test myv_jr-ud-test_2_5.conllu 												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang glv --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/glv_eng_eflomal_gdfa_all.conllu --test gv_cadhan-ud-test_2_7.conllu 												
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal --lang mar --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/mar_eng_eflomal_gdfa_all.conllu --test mr_ufal-ud-test_2_5.conllu												
#python3 pos_tagger_xlmr.py --bronze eflomal --lang yor --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/yor_eng_eflomal_gdfa_all.conllu --test  yo_ytb-ud-test_2_5.conllu												

#! /bin/bash
echo "Eflomal all"
export CUDA_VISIBLE_DEVICES=0,1,2

# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang hun --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/hun_eng_eflomal_gdfa_all.conllu --test hu_szeged-ud-test.conllu > out_eng_hun.txt &
# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang ell --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/ell_eng_eflomal_gdfa_all.conllu --test el_gdt-ud-test.conllu > out_eng_ell.txt &
# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang ces --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/ces_eng_eflomal_gdfa_all.conllu --test cs_cac-ud-test.conllu > out_eng_ces.txt &



# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang hin --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/hin_all_eflomal_gdfa_all.conllu --test hi_hdtb-ud-test_2_5.conllu > out_all_hin.txt &
# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang por --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/por_all_eflomal_gdfa_all.conllu --test  pt_bosque-ud-test_2_5.conllu > out_all_por.txt &
# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang ind --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/ind_all_eflomal_gdfa_all.conllu --test id_gsd-ud-test_2_5.conllu > out_all_ind.txt &
# wait
# python3 pos_tagger_xlmr.py --bronze eflomal_all --lang tur --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/tur_all_eflomal_gdfa_all.conllu --test tr_imst-ud-test_2_5.conllu > out_all_tur.txt &
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang pes --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/pes_all_eflomal_gdfa_all.conllu --test fa_seraji-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang afr --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/afr_all_eflomal_gdfa_all.conllu --test af_afribooms-ud-test_2_5.conllu
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang amh --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/amh_all_eflomal_gdfa_all.conllu --test am_att-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang eus --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/eus_all_eflomal_gdfa_all.conllu --test eu_bdt-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang bul --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/bul_all_eflomal_gdfa_all.conllu --test bg_btb-ud-test_2_5.conllu
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang lit --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/lit_all_eflomal_gdfa_all.conllu --test lt_alksnis-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang tel --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/tel_all_eflomal_gdfa_all.conllu --test te_mtg-ud-test_2_5.conllu 
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang bam --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/bam_all_eflomal_gdfa_all.conllu --test bm_crb-ud-test_2_5.conllu 
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang bel --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/bel_all_eflomal_gdfa_all.conllu --test be_hse-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang myv --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/myv_all_eflomal_gdfa_all.conllu --test myv_jr-ud-test_2_5.conllu 
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang glv --gpu 2 --train /mounts/work/silvia/POS/eflomal/prova/glv_all_eflomal_gdfa_all.conllu --test gv_cadhan-ud-test_2_7.conllu 
#wait
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang mar --gpu 0 --train /mounts/work/silvia/POS/eflomal/prova/mar_all_eflomal_gdfa_all.conllu --test mr_ufal-ud-test_2_5.conllu
#python3 pos_tagger_xlmr.py --bronze eflomal_all --lang yor --gpu 1 --train /mounts/work/silvia/POS/eflomal/prova/yor_all_eflomal_gdfa_all.conllu --test  yo_ytb-ud-test_2_5.conllu > out_all_yor.txt


# python3 pos_tagger_xlmr.py --bronze eflomal_all --epochs 30 --lang amh --gpu 6 --train /mounts/work/silvia/POS/eflomal/prova/amh_eng_eflomal_gdfa_all_new.conllu --test /nfs/datx/UD/v2_5/am_att-ud-test_2_5.conllu > eflomal_eng_amh.txt &
python3 pos_tagger_xlmr.py --bronze eflomal_all --epochs 30 --lang amh_baseline --gpu 7 --train /mounts/work/silvia/POS/data/baseline_data/AMH-FRA-bible-POSUD-TRAIN.txt.conllu --test /nfs/datx/UD/v2_5/am_att-ud-test_2_5.conllu > ramy_amh.txt


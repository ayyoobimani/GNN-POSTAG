#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bronze=1
path="/mounts/work/silvia/POS/filter/"
declare -A tests
# tests=( ["afr"]="af_afribooms-ud-test_2_5.conllu" ["amh"]="am_att-ud-test_2_5.conllu" ["eus"]="eu_bdt-ud-test_2_5.conllu"
#         ["bul"]="bg_btb-ud-test_2_5.conllu" ["hin"]="hi_hdtb-ud-test_2_5.conllu" ["ind"]="id_gsd-ud-test_2_5.conllu"
#         ["lit"]="lt_alksnis-ud-test_2_5.conllu" ["pes"]="fa_seraji-ud-test_2_5.conllu" ["por"]="pt_bosque-ud-test_2_5.conllu"
#         ["tel"]="te_mtg-ud-test_2_5.conllu" ["tur"]="tr_imst-ud-test_2_5.conllu" ["bam"]="bm_crb-ud-test_2_5.conllu"
#         ["bel"]="be_hse-ud-test_2_5.conllu" ["myv"]="myv_jr-ud-test_2_5.conllu" ["glv"]="gv_cadhan-ud-test_2_7.conllu"
#         ["mar"]="mr_ufal-ud-test_2_5.conllu" ["yor"]="yo_ytb-ud-test_2_5.conllu" 
#         ["hun"]="hu_szeged-ud-test.conllu" ["ces"]="cs_cac-ud-test.conllu" ["ell"]="el_gdt-ud-test.conllu")
tests=( ["afr"]="af_afribooms-ud-test_2_5.conllu" ["amh"]="am_att-ud-test_2_5.conllu_nomulti" ["eus"]="eu_bdt-ud-test_2_5.conllu"
        ["bul"]="bg_btb-ud-test_2_5.conllu" ["hin"]="hi_hdtb-ud-test_2_5.conllu" ["ind"]="id_gsd-ud-test_2_5.conllu"
        ["lit"]="lt_alksnis-ud-test_2_5.conllu" ["pes"]="fa_seraji-ud-test_2_5.conllu_nomulti" ["por"]="pt_bosque-ud-test_2_5.conllu_nomulti"
        ["tel"]="te_mtg-ud-test_2_5.conllu" ["tur"]="tr_imst-ud-test_2_5.conllu_nomulti" ["bam"]="bm_crb-ud-test_2_5.conllu"
        ["bel"]="be_hse-ud-test_2_5.conllu" ["myv"]="myv_jr-ud-test_2_5.conllu_nomulti" ["glv"]="gv_cadhan-ud-test_2_7.conllu_nomulti"
        ["mar"]="mr_ufal-ud-test_2_5.conllu_nomulti" ["yor"]="yo_ytb-ud-test_2_5.conllu" 
        ["hun"]="hu_szeged-ud-test.conllu" ["ces"]="cs_cac-ud-test.conllu" ["ell"]="el_gdt-ud-test.conllu")
echo "Our bronze $bronze"

echo "Create train"
#python3 create_train.py --bronze 1 --bible afr-x-bible-newworld.txt --lang afr --thr 0 --bronze_file /mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_afr-x-bible-newworld_15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.8_TypchckTrue_TgBsdSlctTrue_tstamtall_20220511-002256_ElyStpDlta0-GA-chnls1024_small_arb.pickle &
#python3 create_train.py --bronze 1 --bible eus-x-bible-navarrolabourdin.txt --lang eus --thr 0 --bronze_file /mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_eus-x-bible-navarrolabourdin_15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.8_TypchckTrue_TgBsdSlctTrue_tstamtall_20220511-002256_ElyStpDlta0-GA-chnls1024_small_arb.pickle &
#python3 create_train.py --bronze 1 --bible hin-x-bible-bsi.txt --lang hin --thr 0 --bronze_file /mounts/work/ayyoob/results/gnn_align/yoruba/POSTgs_hin-x-bible-bsi_15lngs-POSFeatFalsealltgts_trnsfrmrFalse6LResTrue_trainWEFalse_mskLngTrue_E1_traintgt0.8_TypchckTrue_TgBsdSlctTrue_tstamtall_20220511-002256_ElyStpDlta0-GA-chnls1024_small_arb.pickle

# echo "Start training"
# for VARIABLE in 1 2 3
# do
#     echo $VARIABLE
# 	echo "hin"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang hin --gpu 5 --train /mounts/work/silvia/POS/filter/hin_bronze1_0.0.conllu --test hi_hdtb-ud-test_2_5.conllu >> out_our_hin.txt &
#     P1=$!
#     echo "afr"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang afr --gpu 6 --train /mounts/work/silvia/POS/filter/afr_bronze1_0.0.conllu --test af_afribooms-ud-test_2_5.conllu >> out_our_afr.txt &
#     P2=$!
#     echo "eus"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang eus --gpu 7 --train /mounts/work/silvia/POS/filter/eus_bronze1_0.0.conllu --test eu_bdt-ud-test_2_5.conllu >> out_our_eus.txt &
#     P3=$!
#     wait $P1 $P2 $P3
# done

# echo "Start training"
# # lang1="afr"
# # lang1="amh"
# # lang2="eus"
# # lang2="bul"
# # lang3="ind"
# # lang4="lit"
# # lang3="pes"
# # lang5="hin"

# # lang6="por"
# # lang4="tel"
# # lang7="tur"
# # lang5="bel"
# # lang6="mar"

# lang3="bam"
# lang1="myv"
# lang2="glv"
# # lang7="yor"

# # lang1="hun"
# # lang2="ell"
# # lang3="ces"


# # lang1="amh"
# # lang2="pes"
# # lang3="por"
# # lang1="tur"
# # lang5="myv"
# # lang6="glv"


# # bronze="3"
# # bronze="1"
# # bronze="3eng"
# epochs=15
# # epochs=30
# # WITH XLMR
# for VARIABLE in 1 2 3
# do
#     echo $VARIABLE
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test ${tests[$lang1]} >> out_our${bronze}_${lang1}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test ${tests[$lang1]} >> out_our${bronze}_${lang1}.txt &
#     P1=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${tests[$lang2]} >> out_our${bronze}_${lang2}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${tests[$lang2]} >> out_our${bronze}_${lang2}.txt &
#     P2=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 7 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${tests[$lang3]} >> out_our${bronze}_${lang3}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 7 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${tests[$lang3]} >> out_our${bronze}_${lang3}.txt &
#     P3=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${tests[$lang4]} >> out_our${bronze}_${lang4}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${tests[$lang4]} >> out_our${bronze}_${lang4}.txt &
#     # P4=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${tests[$lang5]} >> out_our${bronze}_${lang5}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${tests[$lang5]} >> out_our${bronze}_${lang5}.txt &
#     # P5=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${tests[$lang6]} >> out_our${bronze}_${lang6}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${tests[$lang6]} >> out_our${bronze}_${lang6}.txt &
#     # P6=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_our${bronze}_${lang7}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_our${bronze}_${lang7}.txt &
#     # P7=$!
#     wait $P1 $P2 $P3 #$P4 $P5 $P6 $P7
#     # wait $P1 $P2
#     # wait $P1 $P2 $P3 $P4
# done

# NO XLMR

# for VARIABLE in 1 2 3
# # for VARIABLE in 1
# do
#     echo $VARIABLE
# 	echo $lang1
#     echo "python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang1 --gpu 3 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test ${tests[$lang1]} >> out_our_${lang1}.txt"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang1 --gpu 3 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test ${tests[$lang1]} >> out_our_${lang1}.txt &
#     P1=$!
#     echo $lang2
#     echo "python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang2 --gpu 4 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${tests[$lang2]} >> out_our_${lang2}.txt"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang2 --gpu 4 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${tests[$lang2]} >> out_our_${lang2}.txt &
#     P2=$!
#     echo $lang3
#     echo "python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang3 --gpu 5 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${tests[$lang3]} >> out_our_${lang3}.txt"
#     python3 pos_tagger_xlmr.py --bronze $bronze --lang $lang3 --gpu 5 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${tests[$lang3]} >> out_our_${lang3}.txt &
#     P3=$!
#     wait $P1 $P2 $P3
#     # wait $P2 $P3
# done

# path="/mounts/work/silvia/POS/eflomal/prova/"
# bronze="eflomal_all"
# # bronze="eflomal_eng"
# epochs=6
# # label="eng"
# label="all"
# for VARIABLE in 1
# do
#     echo $VARIABLE
# 	echo $lang1
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang1]} >> out_${label}_${lang1}_nomulti.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang1]} >> out_${label}_${lang1}_nomulti.txt &
#     P1=$!
    # echo $lang2
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 2 --train ${path}${lang2}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang2]} >> out_${label}_${lang2}_nomulti.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 2 --train ${path}${lang2}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang2]} >> out_${label}_${lang2}_nomulti.txt &
    # P2=$!
    # echo $lang3
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 4 --train ${path}${lang3}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang3]} >> out_${label}_${lang3}_nomulti.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 4 --train ${path}${lang3}_${label}_eflomal_gdfa_all.conllu --test ${tests[$lang3]} >> out_${label}_${lang3}_nomulti.txt 
    # P3=$!
    # wait $P1 $P2 $P3
# done


# exit 0

echo "Start training"
# lang1="afr"
# lang2="bul"
# lang3="ind"

# lang1="amh"
# lang2="eus"
# lang4="lit"
# lang3="pes"
# lang5="hin"
# lang6="por"
# lang7="tel"
# lang1="tur"
# lang2="bel"
# lang3="mar"
# lang4="bam"
# lang5="myv"
# lang6="glv"
# lang7="yor"

# lang1="amh"
# lang2="pes"
# lang3="por"
# lang1="tur"
# lang5="myv"
# lang6="glv"


# bronze="3"

# epochs=15
# # epochs=30
# thr=0.75
# # WITH XLMR
# for VARIABLE in 1 
# do
#     echo $VARIABLE
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_${thr}.conllu --test ${tests[$lang1]} >> out_our${bronze}_${thr}_${lang1}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_${thr}.conllu --test ${tests[$lang1]} >> out_our${bronze}_${thr}_${lang1}.txt &
#     # P1=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_${thr}.conllu --test ${tests[$lang2]} >> out_our${bronze}_${thr}_${lang2}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_${thr}.conllu --test ${tests[$lang2]} >> out_our${bronze}_${thr}_${lang2}.txt &
#     # P2=$!
#     # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}${lang3}_bronze${bronze}_${thr}.conllu --test ${tests[$lang3]} >> out_our${bronze}_${thr}_${lang3}.txt"
#     # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}${lang3}_bronze${bronze}_${thr}.conllu --test ${tests[$lang3]} >> out_our${bronze}_${thr}_${lang3}.txt &
#     # P3=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_${thr}.conllu --test ${tests[$lang4]} >> out_our${bronze}_noxlmr_${thr}_${lang4}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_${thr}.conllu --test ${tests[$lang4]} >> out_our${bronze}_noxlmr_${thr}_${lang4}.txt &
#     P4=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 7 --train ${path}noxlmr_${lang5}_bronze${bronze}_${thr}.conllu --test ${tests[$lang5]} >> out_our${bronze}_noxlmr_${thr}_${lang5}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 7 --train ${path}noxlmr_${lang5}_bronze${bronze}_${thr}.conllu --test ${tests[$lang5]} >> out_our${bronze}_noxlmr_${thr}_${lang5}.txt &
#     P5=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}noxlmr_${lang6}_bronze${bronze}_${thr}.conllu --test ${tests[$lang6]} >> out_our${bronze}_noxlmr_${thr}_${lang6}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}noxlmr_${lang6}_bronze${bronze}_${thr}.conllu --test ${tests[$lang6]} >> out_our${bronze}_noxlmr_${thr}_${lang6}.txt &
#     P6=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}noxlmr_${lang7}_bronze${bronze}_${thr}.conllu --test ${tests[$lang7]} >> out_our${bronze}_noxlmr_${thr}_${lang7}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}noxlmr_${lang7}_bronze${bronze}_${thr}.conllu --test ${tests[$lang7]} >> out_our${bronze}_noxlmr_${thr}_${lang7}.txt &
#     P7=$!
#     wait $P4 $P5 $P6 $P7 # $P1 $P2 $P3 #
#     # wait $P1 $P2
#     # wait $P1 $P2 $P3 $P4
# done

path_test='/nfs/datx/UD/v2_5/'

bronze="3"
# lang1="afr"
# lang2="bul"
# lang3="ind"
# lang4="amh"
# lang5="eus"
# lang6="lit"
# lang7="pes"

lang6="hin"
lang5="por"
lang4="tel"
lang1="tur"
lang2="bel"
lang3="mar"

# epochs=15
epochs=30
thr=0.0
# NO XLMR
for VARIABLE in 1
do
    echo $VARIABLE
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}noxlmr_${lang1}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang1]} >> out_our${bronze}_noxlmr_${thr}_${lang1}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}noxlmr_${lang1}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang1]} >> out_our${bronze}_noxlmr_${thr}_${lang1}.txt &
    P1=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}noxlmr_${lang2}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang2]} >> out_our${bronze}_noxlmr_${thr}_${lang2}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}noxlmr_${lang2}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang2]} >> out_our${bronze}_noxlmr_${thr}_${lang2}.txt &
    P2=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang3]} >> out_our${bronze}_noxlmr_${thr}_${lang3}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang3]} >> out_our${bronze}_noxlmr_${thr}_${lang3}.txt &
    P3=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang4]} >> out_our${bronze}_noxlmr_${thr}_${lang4}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang4]} >> out_our${bronze}_noxlmr_${thr}_${lang4}.txt &
    P4=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}noxlmr_${lang5}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang5]} >> out_our${bronze}_noxlmr_${thr}_${lang5}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}noxlmr_${lang5}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang5]} >> out_our${bronze}_noxlmr_${thr}_${lang5}.txt &
    P5=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}noxlmr_${lang6}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang6]} >> out_our${bronze}_noxlmr_${thr}_${lang6}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}noxlmr_${lang6}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang6]} >> out_our${bronze}_noxlmr_${thr}_${lang6}.txt &
    P6=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}noxlmr_${lang7}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang7]} >> out_our${bronze}_noxlmr_${thr}_${lang7}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}noxlmr_${lang7}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang7]} >> out_our${bronze}_noxlmr_${thr}_${lang7}.txt &
    # P7=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang8 --gpu 7 --train ${path}noxlmr_${lang8}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang8]} >> out_our${bronze}_noxlmr_${thr}_${lang8}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang8 --gpu 7 --train ${path}noxlmr_${lang8}_bronze${bronze}_${thr}.conllu --test ${path_test}${tests[$lang8]} >> out_our${bronze}_noxlmr_${thr}_${lang8}.txt &
    # P8=$!
    wait $P4 $P5 $P6 $P1 $P2 $P3 #
    # wait $P1 $P2
    # wait $P1 $P2 $P3 $P4
done


# python3 pos_tagger_xlmr.py --epochs 15 --bronze 3 --lang afr --gpu 0 --train /mounts/work/silvia/POS/filter/afr_bronze3_0.0.conllu --test /nfs/datx/UD/v2_5/af_afribooms-ud-test_2_5.conllu

#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bronze=1
path="/mounts/work/silvia/POS/filter/"
declare -A tests
tests=( ["afr"]="UD_Afrikaans-AfriBooms/af_afribooms-ud-test.conllu" 
        ["amh"]="UD_Amharic-ATT/am_att-ud-test.conllu_nomulti"  #####
        ["eus"]="UD_Basque-BDT/eu_bdt-ud-test.conllu"
        ["bul"]="UD_Bulgarian-BTB/bg_btb-ud-test.conllu" 
        ["hin"]="UD_Hindi-HDTB/hi_hdtb-ud-test.conllu" 
        ["ind"]="UD_Indonesian-GSD/id_gsd-ud-test.conllu_nomulti" #######
        ["lit"]="UD_Lithuanian-ALKSNIS/lt_alksnis-ud-test.conllu" 
        ["pes"]="UD_Persian-Seraji/fa_seraji-ud-test.conllu_nomulti" ####
        ["por"]="UD_Portuguese-Bosque/pt_bosque-ud-test.conllu_nomulti" ###
        ["tel"]="UD_Telugu-MTG/te_mtg-ud-test.conllu" 
        ["tur"]="UD_Turkish-IMST/tr_imst-ud-test.conllu_nomulti" #####
        ["bam"]="UD_Bambara-CRB/bm_crb-ud-test.conllu"
        ["bel"]="UD_Belarusian-HSE/be_hse-ud-test.conllu" 
        ["myv"]="UD_Erzya-JR/myv_jr-ud-test.conllu_nomulti" ###
        ["glv"]="UD_Manx-Cadhan/gv_cadhan-ud-test.conllu_nomulti" ###
        ["mar"]="UD_Marathi-UFAL/mr_ufal-ud-test.conllu_nomulti" ####
        ["yor"]="UD_Yoruba-YTB/yo_ytb-ud-test.conllu_nomulti" ####
        ["hun"]="UD_Hungarian-Szeged/hu_szeged-ud-test.conllu" 
        ["ces"]="UD_Czech-CAC/cs_cac-ud-test.conllu" 
        ["ell"]="UD_Greek-GDT/el_gdt-ud-test.conllu" ) 
echo "Our bronze $bronze"


# echo "Start training"
# lang1="afr"
# lang1="amh"
# lang2="eus"
# lang3="bul"
# lang6="ind"
# lang5="lit"
# lang6="pes"
# lang7="hin"
# lang2="por"

# # lang8="tel"
# lang3="tur"
# lang4="bel"
# lang5="mar"

# lang1="bam"
# lang2="myv"
# lang3="glv"
lang1="yor"

# lang1="hun"
# lang2="ell"
# lang3="ces"


# lang1="amh"
# lang2="pes"
# lang3="por"
# lang1="tur"
# lang5="myv"
# lang6="glv"


bronze="3"
test_path="/nfs/datx/UD/v2_10/ud-treebanks-v2.10/"
epochs=15
# WITH XLMR
for VARIABLE in 1 2 3
do
    echo $VARIABLE
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang1]} >> 2v10_out_our${bronze}_${lang1}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang1]} >> 2v10_out_our${bronze}_${lang1}.txt &
    P1=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang2]} >> 2v10_out_our${bronze}_${lang2}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang2]} >> 2v10_out_our${bronze}_${lang2}.txt &
#     P2=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang3]} >> 2v10_out_our${bronze}_${lang3}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang3]} >> 2v10_out_our${bronze}_${lang3}.txt &
#     P3=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_${lang4}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_${lang4}.txt &
#     P4=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang5]} >> 2v10_out_our${bronze}_${lang5}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang5]} >> 2v10_out_our${bronze}_${lang5}.txt &
    # P5=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang6]} >> 2v10_out_our${bronze}_${lang6}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang6]} >> 2v10_out_our${bronze}_${lang6}.txt &
    # P6=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang7]} >> 2v10_out_our${bronze}_${lang7}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang7]} >> 2v10_out_our${bronze}_${lang7}.txt &
    # P7=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang8 --gpu 7 --train ${path}${lang8}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang8]} >> 2v10_out_our${bronze}_${lang8}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang8 --gpu 7 --train ${path}${lang8}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang8]} >> 2v10_out_our${bronze}_${lang8}.txt &
    # P8=$!
    # wait $P1 $P2 $P3 $P4 #$P5 $P6 $P7 $P8
    # wait $P1 $P2 $P3
    # wait $P1 $P2 $P3 $P4
    wait $P1
done

# NO XLMR

# lang1="afr"
# lang2="amh"
# lang3="eus"
# lang4="bul"
# lang5="ind"
# lang6="lit"
# lang7="pes"
# lang8="hin"

# lang1="por"
# lang2="tel"
# lang3="tur"
# lang4="bel"
# lang5="mar"


# epochs=30
# bronze="3"
# test_path="/nfs/datx/UD/v2_10/ud-treebanks-v2.10/"
# for VARIABLE in 1 2 3
# do
#     echo $VARIABLE
# 	echo $lang1
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang1]} >> 2v10_out_our${bronze}_noxlmr_${lang1}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 0 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang1]} >> 2v10_out_our${bronze}_noxlmr_${lang1}.txt &
#     P1=$!
#     echo $lang2
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang2]} >> 2v10_out_our${bronze}_noxlmr_${lang2}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 1 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang2]} >> 2v10_out_our${bronze}_noxlmr_${lang2}.txt &
#     P2=$!
#     echo $lang3
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang3]} >> 2v10_out_our${bronze}_noxlmr_${lang3}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang3]} >> 2v10_out_our${bronze}_noxlmr_${lang3}.txt &
#     P3=$!
#     echo $lang4
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_noxlmr_${lang4}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}noxlmr_${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_noxlmr_${lang4}.txt &
#     P4=$!
    # echo $lang5
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 4 --train ${path}noxlmr_${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_noxlmr_${lang4}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 4 --train ${path}noxlmr_${lang4}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang4]} >> 2v10_out_our${bronze}_noxlmr_${lang4}.txt &
    # P5=$!
    # echo $lang6
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 5 --train ${path}noxlmr_${lang5}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang5]} >> 2v10_out_our${bronze}_noxlmr_${lang5}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 5 --train ${path}noxlmr_${lang5}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang5]} >> 2v10_out_our${bronze}_noxlmr_${lang5}.txt &
    # P6=$!
    # echo $lang3
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 6 --train ${path}noxlmr_${lang6}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang6]} >> 2v10_out_our${bronze}_noxlmr_${lang6}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 6 --train ${path}noxlmr_${lang6}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang6]} >> 2v10_out_our${bronze}_noxlmr_${lang6}.txt &
    # P7=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 7 --train ${path}noxlmr_${lang7}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang7]} >> 2v10_out_our${bronze}_noxlmr_${lang7}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 7 --train ${path}noxlmr_${lang7}_bronze${bronze}_0.0.conllu --test ${test_path}${tests[$lang7]} >> 2v10_out_our${bronze}_noxlmr_${lang7}.txt &
    # P8=$!
    # wait $P1 $P2 $P3 $P4 #$P5 $P6 $P7 $P8
    # wait $P2 $P3
# done



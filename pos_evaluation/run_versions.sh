#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

path="/mounts/work/silvia/POS/filter/"
declare -A tests
# tests=( ["afr"]="af_afribooms-ud-test_2_5.conllu" ["amh"]="am_att-ud-test_2_5.conllu" ["eus"]="eu_bdt-ud-test_2_5.conllu"
#         ["bul"]="bg_btb-ud-test_2_5.conllu" ["hin"]="hi_hdtb-ud-test_2_5.conllu" ["ind"]="id_gsd-ud-test_2_5.conllu"
#         ["lit"]="lt_alksnis-ud-test_2_5.conllu" ["pes"]="fa_seraji-ud-test_2_5.conllu" ["por"]="pt_bosque-ud-test_2_5.conllu"
#         ["tel"]="te_mtg-ud-test_2_5.conllu" ["tur"]="tr_imst-ud-test_2_5.conllu" ["bam"]="bm_crb-ud-test_2_5.conllu"
#         ["bel"]="be_hse-ud-test_2_5.conllu" ["myv"]="myv_jr-ud-test_2_5.conllu" ["glv"]="gv_cadhan-ud-test_2_7.conllu"
#         ["mar"]="mr_ufal-ud-test_2_5.conllu" ["yor"]="yo_ytb-ud-test_2_5.conllu" 
#         ["hun"]="hu_szeged-ud-test.conllu" ["ces"]="cs_cac-ud-test.conllu" ["ell"]="el_gdt-ud-test.conllu")
# tests2_5=( ["afr"]="af_afribooms-ud-test_2_5.conllu" ["amh"]="am_att-ud-test_2_5.conllu_nomulti" ["eus"]="eu_bdt-ud-test_2_5.conllu"
#         ["bul"]="bg_btb-ud-test_2_5.conllu" ["hin"]="hi_hdtb-ud-test_2_5.conllu" ["ind"]="id_gsd-ud-test_2_5.conllu"
#         ["lit"]="lt_alksnis-ud-test_2_5.conllu" ["pes"]="fa_seraji-ud-test_2_5.conllu_nomulti" ["por"]="pt_bosque-ud-test_2_5.conllu_nomulti"
#         ["tel"]="te_mtg-ud-test_2_5.conllu" ["tur"]="tr_imst-ud-test_2_5.conllu_nomulti" ["bam"]="bm_crb-ud-test_2_5.conllu"
#         ["bel"]="be_hse-ud-test_2_5.conllu" ["myv"]="myv_jr-ud-test_2_5.conllu_nomulti" ["glv"]="gv_cadhan-ud-test_2_7.conllu_nomulti"
#         ["mar"]="mr_ufal-ud-test_2_5.conllu_nomulti" ["yor"]="yo_ytb-ud-test_2_5.conllu" 
#         ["hun"]="hu_szeged-ud-test.conllu" ["ces"]="cs_cac-ud-test.conllu" ["ell"]="el_gdt-ud-test.conllu")

# path_1_2="/nfs/datx/UD/v1_2/universal-dependencies-1.2/"
# tests=( ["bul"]="UD_Bulgarian//bg-ud-test.conllu" ["hin"]="UD_Hindi/hi-ud-test.conllu" ["ind"]="UD_Indonesian/id-ud-test.conllu"
#        ["pes"]="UD_Persian/fa-ud-test.conllu_nomulti" ["por"]="UD_Portuguese/pt-ud-test.conllu")

# path_2_0="/nfs/datx/UD/v2_0/"
# tests=( ["bul"]="bg-ud-test_2_0.conllu" ["por"]="pt-ud-test_2_0.conllu_nomulti")

path_2_3="/nfs/datx/UD/v2_3/"
tests=( ["afr"]="af_afribooms-ud-test_2_3.conllu" ["amh"]="am_att-ud-test_2_3.conllu_nomulti" ["bam"]="bm_crb-ud-test_2_3.conllu"
        ["ind"]="id_pud-ud-test_2_3.conllu" ["hin"]="hi_pud-ud-test_2_3.conllu" ["por"]="pt_pud-ud-test_2_3.conllu_nomulti"
        ["myv"]="myv_jr-ud-test_2_3.conllu_nomulti" ["tur"]="tr_pud-ud-test_2_3.conllu_nomulti" ["yor"]="yo_ytb-ud-test_2_3.conllu"
        )
	     

echo "Our bronze $bronze"


echo "Start training"
# lang1="afr"
# lang2="amh"
# lang1=pes
# lang1="bul"
# lang3="ind"
# lang4="hin"
# lang5="por"
# lang1="tur"
lang1="bam"
lang2="myv"
# lang3="yor"


# bronze="1"
bronze="3"
# version="1_2"
# version="2_0"
version="2_3"
# path_version=$path_1_2
# path_version=$path_2_0
path_version=$path_2_3

# bronze="1eng"
# epochs=30
# epochs=15
# WITH XLMR
# for VARIABLE in 1 2 3
# do
#     echo $VARIABLE
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 5 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test  ${path_version}${tests[$lang1]} >> ${version}_out_our${bronze}_${lang1}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 5 --train ${path}${lang1}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang1]} >> ${version}_out_our${bronze}_${lang1}.txt &
#     P1=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 6 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang2]} >> ${version}_out_our${bronze}_${lang2}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 6 --train ${path}${lang2}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang2]} >> ${version}_out_our${bronze}_${lang2}.txt &
#     P2=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 7 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang3]} >> ${version}_out_our${bronze}_${lang3}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 7 --train ${path}${lang3}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang3]} >> ${version}_out_our${bronze}_${lang3}.txt &
#     P3=$!
#     echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 4 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang4]} >> ${version}_out_our${bronze}_${lang4}.txt"
#     python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 4 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang4]} >> ${version}_out_our${bronze}_${lang4}.txt &
#     P4=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang5]} >> ${version}_out_our${bronze}_${lang5}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang5]} >> ${version}_out_our${bronze}_${lang5}.txt &
    # P5=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang6]} >> ${version}_out_our${bronze}_${lang6}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang6]} >> ${version}_out_our${bronze}_${lang6}.txt &
    # P6=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_oureng_${lang7}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_oureng_${lang7}.txt &
    # P7=$!
    # wait $P1 #$P2 #$P3 $P4 $P5 $P6
#     wait $P1 $P2 $P3 $P4 #$P5
# done


# NOXLMR
epochs=30
for VARIABLE in 1 2 3
do
    echo $VARIABLE
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 6 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test  ${path_version}${tests[$lang1]} >> ${version}_noxlmr_out_our${bronze}_${lang1}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang1 --gpu 6 --train ${path}noxlmr_${lang1}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang1]} >> ${version}_noxlmr_out_our${bronze}_${lang1}.txt &
    P1=$!
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 7 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang2]} >> ${version}_noxlmr_out_our${bronze}_${lang2}.txt"
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang2 --gpu 7 --train ${path}noxlmr_${lang2}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang2]} >> ${version}_noxlmr_out_our${bronze}_${lang2}.txt &
    P2=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang3]} >> ${version}_noxlmr_out_our${bronze}_${lang3}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang3 --gpu 2 --train ${path}noxlmr_${lang3}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang3]} >> ${version}_noxlmr_out_our${bronze}_${lang3}.txt &
    # P3=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang4]} >> ${version}_out_our${bronze}_${lang4}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang4 --gpu 3 --train ${path}${lang4}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang4]} >> ${version}_out_our${bronze}_${lang4}.txt &
    # P4=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang5]} >> ${version}_out_our${bronze}_${lang5}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang5 --gpu 4 --train ${path}${lang5}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang5]} >> ${version}_out_our${bronze}_${lang5}.txt &
    # P5=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang6]} >> ${version}_out_our${bronze}_${lang6}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang6 --gpu 5 --train ${path}${lang6}_bronze${bronze}_0.0.conllu --test ${path_version}${tests[$lang6]} >> ${version}_out_our${bronze}_${lang6}.txt &
    # P6=$!
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_oureng_${lang7}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang7 --gpu 6 --train ${path}${lang7}_bronze${bronze}_0.0.conllu --test ${tests[$lang7]} >> out_oureng_${lang7}.txt &
    # P7=$!
    # wait $P1 $P2 $P3 $P4 $P5 $P6
    wait $P1 $P2 #$P3
done
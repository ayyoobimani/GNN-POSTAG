export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

path="/mounts/work/silvia/POS/filter/"
declare -A tests
echo "Start training"
tests=( ["bam"]="bm_crb-ud-test_2_5.conllu"
        ["myv"]="myv_jr-ud-test_2_5.conllu_nomulti" ["glv"]="gv_cadhan-ud-test_2_7.conllu_nomulti")


lang="bam"
# lang="myv"
# lang="glv"
bronze="3"
gpu=0

epochs=30
thr=0.0
# WITHOUT XLMR
for VARIABLE in 1 2
do
    echo $VARIABLE
    # echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang --gpu $gpu --train ${path}noxlmr_${lang}_bronze${bronze}_${thr}.conllu --test ${tests[$lang]} >> wang_noxlmr_${bronze}_${thr}_${lang}.txt"
    echo "python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang --gpu $gpu --train ${path}noxlmr_${lang}_bronze${bronze}_${thr}.conllu --test ${tests[$lang]} >> wang_after_noxlmr_${bronze}_${thr}_${lang}.txt"
    # python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang --gpu $gpu --train ${path}noxlmr_${lang}_bronze${bronze}_${thr}.conllu --test ${tests[$lang]} >> wang_noxlmr_${bronze}_${thr}_${lang}.txt 
    python3 pos_tagger_xlmr.py --epochs $epochs --bronze $bronze --lang $lang --gpu $gpu --train ${path}noxlmr_${lang}_bronze${bronze}_${thr}.conllu --test ${tests[$lang]} >> wang_after_noxlmr_${bronze}_${thr}_${lang}.txt 

done
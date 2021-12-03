export FLASK_APP=align.py
export FLASK_SECRET_KEY="ddddddddddddddddd"
export FLASK_RUN_PORT=8000
export CONFIG_PATH="config_pbc.ini"
#conda activate align_graph

python -um app.alignment_graph_producer 
#--model_save_path $MODEL_SAVE_PATH \
#--save_path $SAVE_PATH \
#--gold_file $GOLD_FILE \
#--base_align_method $BASE_ALIGNMENT_METHOD \
#--alignments_path $ALIGNMENTS_PATH \
#--source_edition $SOURCE_EDITION \
#--target_edition $TARGET_EDITION \
#--editions_file $EDITIONS_FILE \
#--edition_count $EDITION_COUNT \
#--core_count $CORE_COUNT \

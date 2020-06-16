#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 python bert/run_classifier.py \
	--task_name=Emotion \
	--do_predict=true \
	--data_dir=$DATA_DIR \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=output/distill/model.ckpt-624 \
	--output_dir=output/distill

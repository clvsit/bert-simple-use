export MODEL_DIR=/nfs/users/chenxu/project/bert_repeat_checker/model/chinese_L-12_H-768_A-12

CUDA_VISIBLE_DEVICES=1 python bert/run_classifier.py \
	--data_dir=data \
	--task_name=Emotion \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=output/baseline/model.ckpt-624 \
	--do_export=true \
	--output_dir=output/baseline \
	--export_dir=exported

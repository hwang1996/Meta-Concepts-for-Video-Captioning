CUDA_VISIBLE_DEVICES=0 python misc/train.py \
			--checkpoint_path output/model/run \
			--model_file output/model/run/model.pth \
			--best_model_file output/model/run/model_best.pth \
			--result_file output/model/run/test_results.json \
			--start_from output/model/run/model_best.pth \
			--batch_size 32 \
			--rnn_num_layers 1 \
			--learning_rate 8e-5 \
			--lr_update 100 \
			--lr_decay_rate 0.5 \
			--max_epochs 150 \
			--beam_size 5 \
			--use_rl=1 \
			--use_cst=1 \
			--use_mixer=0 \
			--scb_captions=0 \
			--save_checkpoint_every 1 \
			2>&1 | tee output/model/run/log.txt

CUDA_VISIBLE_DEVICES=0 python misc/test.py \
		--result_file output/model/expand/all_rl/test_results.json \
		--model_file output/model/expand/all_rl/model_best.pth \
		--rnn_num_layers 1 \
		--beam_size 5 \
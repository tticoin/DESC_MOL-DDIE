k=5
for i in `seq 1 $k`;do
CUDA_VISIBLE_DEVICES=2 python run_ddie.py \
    --task_name MRPC \
    --model_type bert \
    --data_dir /mnt/analysis/sentlen_label_skf_cv$k/$i/tsv \
    --model_name_or_path /mnt/model/scibert_scivocab_uncased \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 3. \
    --dropout_prob .1 \
    --weight_decay .01 \
    --fp16 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --max_seq_length 128 \
    --use_cnn \
    --conv_window_size 5 \
    --pos_emb_dim 10 \
    --activation gelu \
    --middle_layer_size 0 \
    --use_mol \
    --desc_conv_window_size 3 \
    --desc_conv_output_size 20 \
    --molecular_vector_size 50 \
    --gnn_layer_hidden 5 \
    --gnn_layer_output 1 \
    --gnn_mode sum \
    --gnn_activation gelu \
    --fingerprint_dir /mnt/analysis/sentlen_label_skf_cv$k/$i/rad1 \
    --overwrite_output_dir \
    --output_dir /mnt/DESC_MOL-DDIE/sentlen_label_skf_cv$k/$i/rad1
done

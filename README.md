# DESC\_MOL-DDIE
Implementation of Using Drug Description and Molecular Structures for Drug-Drug Interaction Extraction from Literature

# Requirements
python3  
torch >= 1.2  
transformers == 2.1  
rdkit  
lxml  

# Usage
## Preparation of the corpus sets
see [corpus/README.md](corpus/README.md)

## Preparation of the DrugBank data
see [database/README.md](database/README.md)

## Preparation of the molecular fingerprints data
see [fingerprint/README.md](fingerprint/README.md)

## DDI Extraction
```
cd main
python run_ddie.py \
    --task_name MRPC \
    --model_type bert \
    --data_dir $NEW_TSV_DIR
    --model_name_or_path $SCIBERT_MODEL
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
    --use_desc \
    --desc_conv_window_size 3 \
    --desc_conv_output_size 20 \
    --use_mol \
    --molecular_vector_size 50 \
    --gnn_layer_hidden 5 \
    --gnn_layer_output 1 \
    --gnn_mode sum \
    --gnn_activation gelu \
    --fingerprint_dir $FINGERPRINT_DIR \
    --output_dir $OUTPUT_DIR
```

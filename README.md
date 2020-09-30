# DESC\_MOL-DDIE
Implementation of Using Drug Description and Molecular Structures for Drug-Drug Interaction Extraction from Literature

## Requirements
python3  
torch >= 1.2  
transformers == 2.1  
rdkit  
lxml  

## Usage
### Preparation of the corpus sets
see [corpus/README.md](corpus/README.md)

### Preparation of the DrugBank data
see [database/README.md](database/README.md)

### Preparation of the molecular fingerprints data
see [fingerprint/README.md](fingerprint/README.md)

### Preparation of the SciBERT model
pre-trained SciBERT model is availabel [here](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar)

### Sample data set
When you use the [sample data set](sample) created by splitting the official training data set, you can skip the preparation of the corpus and the database.
```
export $NEW_TSV_DIR=sample/tsv
export $FINGERPRINT_DIR=sample/radius1
export $RADIUS=1
python3 fingerprint/preprocessor.py $NEW_TSV_DIR none $RADIUS $FINGERPRINT_DIR
```

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
    --desc_conv_window_size 3 \
    --desc_conv_output_size 20 \
    --molecular_vector_size 50 \
    --gnn_layer_hidden 5 \
    --gnn_layer_output 1 \
    --gnn_mode sum \
    --gnn_activation gelu \
    --fingerprint_dir $FINGERPRINT_DIR \
    --output_dir $OUTPUT_DIR
```
when you use description and molecular strucuture information, please add ```--use_desc``` and ```--use_mol``` arguments respectively.


## Acknowledgement
This work was supported by JSPS KAKENHI Grant Numbers 17K12741 and 20k11962

mkdir $DICT_DIR
python3 get_dict.py $DRUGBANK_XML $DICT_DIR/entry.dict $DICT_DIR/desc.dict $DICT_DIR/smiles.dict

python3 mention2dbid.py $TSV_DIR/train.tsv $DICT_DIR/entry.dict $DICT_DIR/mention_train.dict
python3 mention2dbid.py $TSV_DIR/dev.tsv $DICT_DIR/entry.dict $DICT_DIR/mention_dev.dict

mkdir $NEW_TSV_DIR
python3 add_db.py $TSV_DIR/train.tsv $DICT_DIR/mention_train.dict $DICT_DIR/smiles.dict $DICT_DIR/desc.dict $NEW_TSV_DIR/train.tsv
python3 add_db.py $TSV_DIR/dev.tsv $DICT_DIR/mention_dev.dict $DICT_DIR/smiles.dict $DICT_DIR/desc.dict $NEW_TSV_DIR/dev.tsv

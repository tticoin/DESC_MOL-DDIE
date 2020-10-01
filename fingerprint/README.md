### Extracting fingerprints from SMILES files and save to numpy files
```
export $FINGERPRINT_DIR=(output npy files dir)
export $RADIUS=(0 or 1 or 2)
python3 preprocessor.py $NEW_TSV_DIR none $RADIUS $FINGERPRINT_DIR
```

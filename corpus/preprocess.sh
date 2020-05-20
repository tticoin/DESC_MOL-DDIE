# Replace space to underscore in filenames
for i in $TARGET_DIR/*.xml; do
    mv "$i" `echo $i | sed -e 's/ /_/g'`
done

# Convert XML to Brat format
mkdir $BRAT_DIR
for i in $TARGET_DIR/*.xml; do
    python3 xml2brat.py $i $BRAT_DIR/`basename $i .xml`
done

# Convert Brat to TSV format
mkdir $TSV_DIR
python3 brat2tsv.py $BRAT_DIR $TSV_DIR/$TSV_NAME

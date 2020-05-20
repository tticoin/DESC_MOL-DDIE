# Usage
## download corpus set
Contact the task organizers and download data set

## conbert XML format to TSV format
```
export XML_DIR=(semeval2013 xml files dir)
export BRAT_DIR=(brat format output files dir)
export TSV_DIR=(tsv format output files dir)

# Replace space to underscore in filenames
for i in $XML_DIR/*.xml; do
    mv "$i" `echo $i | sed -e 's/ /_/g'`
done

# Convert XML to Brat format
mkdir $BRAT_DIR
for i in $XML_DIR/*.xml; do
    python3 xml2brat.py $i $BRAT_DIR/`basename $i .xml`
done

# Convert Brat to TSV format
mkdir $TSV_DIR
python3 brat2tsv.py $BRAT_DIR $TSV_DIR/(train.tsv or dev.tsv)
```

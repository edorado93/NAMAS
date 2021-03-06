#/bin/bash

export AGIGA=$1
export WORK=$2
export THREADS=30
export SCRIPTS=$ABS/dataset
export SPLITS=$ABS/dataset
export UNK=5

echo "Step 1: Construct the title-article pairs from gigaword"
mkdir -p $WORK
find $AGIGA/data -name "*.xml.gz" | parallel --gnu --progress -j $THREADS python2.7 $SCRIPTS/process_agiga.py \{\} $WORK

echo "Step 2: Compile the data into train/dev/test."
cd $WORK
cat $SPLITS/train.splits | sed 's/^/xml\//'  | xargs cat > train.data.txt
cat $SPLITS/valid.splits | sed 's/^/xml\//'  | xargs cat > valid.data.txt
cat $SPLITS/test.splits  | sed 's/^/xml\//'  | xargs cat > test.data.txt


echo "Step 3: Basic filtering on train/dev."
python2.7 $SCRIPTS/filter.py train.data.txt > train.data.filter.txt
python2.7 $SCRIPTS/filter.py valid.data.txt > valid.data.filter.txt


echo "Step 4: Compile dictionary."
python2.7 $SCRIPTS/make_dict.py $WORK/train.data.filter.txt  $WORK/train $UNK


echo "Step 5: Construct title-article files."
python2.7 $SCRIPTS/pull.py trg_lc $WORK/train.title.dict   < $WORK/train.data.filter.txt > $WORK/train.title.txt
python2.7 $SCRIPTS/pull.py src_lc $WORK/train.article.dict < $WORK/train.data.filter.txt > $WORK/train.article.txt

echo "----> Done constructing title-article files on the training data"
python2.7 $SCRIPTS/pull.py trg_lc $WORK/train.title.dict   < $WORK/valid.data.txt > $WORK/valid.title.txt
python2.7 $SCRIPTS/pull.py src_lc $WORK/train.article.dict < $WORK/valid.data.txt > $WORK/valid.article.txt

echo "----> Done constructing title-article files on the validation data"
python2.7 $SCRIPTS/pull.py trg_lc $WORK/train.title.dict   < $WORK/test.data.txt > $WORK/test.title.txt
python2.7 $SCRIPTS/pull.py src_lc $WORK/train.article.dict < $WORK/test.data.txt > $WORK/test.article.txt

echo "----> Done constructing title-article files on the test data"
python2.7 $SCRIPTS/pull.py trg_lc $WORK/train.title.dict   < $WORK/valid.data.filter.txt > $WORK/valid.title.filter.txt
python2.7 $SCRIPTS/pull.py src_lc $WORK/train.article.dict < $WORK/valid.data.filter.txt > $WORK/valid.article.filter.txt

echo "Step 6: Constructing torch data files."
bash $ABS/prep_torch_data.sh $WORK

mkdir -p data/$1/labeled/GT
mkdir -p data/$1/labeled/input
mkdir -p data/$1/labeled/LA

mkdir -p data/$1/unlabeled/input
mkdir -p data/$1/unlabeled/LA
mkdir -p data/$1/unlabeled/candidate

mkdir -p data/$1/val/GT
mkdir -p data/$1/val/input
mkdir -p data/$1/val/LA

ln -s /home/ddimauro/Neptune/Enhancement/Semi-UIR/data/uieb/test data/$1/test
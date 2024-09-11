FILE="HandGesture"

echo "Specified File is [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/datasets/HandGestureRecognition/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
echo "Extracting $TAR_FILE"
echo "Extracting to $TARGET_DIR"
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
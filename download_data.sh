cd data/multiwoz/

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz/MultiWOZ2.1_Cleaned.zip"
unzip -o MultiWOZ2.1_Cleaned.zip
mv MultiWOZ2.1_Cleaned en
rm MultiWOZ2.1_Cleaned.zip

mkdir zh
cd zh

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/human_val_data.zip"
unzip -o human_val_data.zip
rm human_val_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/mt_data.zip"
unzip -o mt_data.zip
rm mt_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/dstc9-test-250.zip"
unzip -o dstc9-test-250.zip
rm dstc9-test-250.zip

wget "https://raw.githubusercontent.com/ConvLab/ConvLab-2/master/data/multiwoz_zh/ontology-data.json"

cd ../../crosswoz/

mkdir zh
cd zh

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/train.json.zip"
unzip -o train.json.zip
rm train.json.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/val.json.zip"
unzip -o val.json.zip
rm val.json.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/test.json.zip"
unzip -o test.json.zip
rm test.json.zip

cd ..
mkdir en
cd en

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz_en/human_val_data.zip"
unzip -o human_val_data.zip
rm human_val_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz_en/mt_data.zip"
unzip -o mt_data.zip
rm mt_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz_en/dstc9-test-250.zip"
unzip -o dstc9-test-250.zip
rm dstc9-test-250.zip

wget "https://raw.githubusercontent.com/ConvLab/ConvLab-2/master/data/crosswoz_en/ontology-data.json"

python preprocess_crosswoz_data.py zh
python preprocess_crosswoz_data.py en
python preprocess_multiwoz_data.py zh
python preprocess_multiwoz_data.py en

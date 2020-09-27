mkdir zh
cd zh

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/train.json.zip"
unzip train.json.zip
rm train.json.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/val.json.zip"
unzip val.json.zip
rm val.json.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz/test.json.zip"
unzip test.json.zip
rm test.json.zip

cd ..
mkdir en
cd en

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz_en/human_val_data.zip"
unzip human_val_data.zip
rm human_val_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/crosswoz_en/mt_data.zip"
unzip mt_data.zip
rm mt_data.zip

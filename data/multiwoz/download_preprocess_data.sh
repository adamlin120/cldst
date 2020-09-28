wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz/MultiWOZ2.1_Cleaned.zip"
unzip MultiWOZ2.1_Cleaned.zip
rm MultiWOZ2.1_Cleaned.zip

mkdir zh
cd zh

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/human_val_data.zip"
unzip human_val_data.zip
rm human_val_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/mt_data.zip"
unzip mt_data.zip
rm mt_data.zip

wget "https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz_zh/dstc9-test-250.zip"
unzip dstc9-test-250.zip
rm dstc9-test-250.zip

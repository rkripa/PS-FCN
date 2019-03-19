(
    mkdir -p data/datasets
    cd data/datasets
    gsutil cp gs://ps-fcn-ds/PS_Blobby_Dataset.tgz .
    tar xzf PS_Blobby_Dataset.tgz
    rm PS_Blobby_Dataset.tgz
)

(
    cd data/datasets
    gsutil cp gs://ps-fcn-ds/DiLiGenT.zip .
    unzip -qq DiLiGenT.zip
    rm DiLiGenT.zip
    cd DiLiGenT/pmsData/
    ls | sed '/objects.txt/d' > objects.txt
    cp ballPNG/filenames.txt .
)


(
    mkdir -p data/models/
    cd data/models
    gsutil cp gs://ps-fcn-md/PS-FCN_B_S_32.pth.tar .
)



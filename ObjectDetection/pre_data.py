import os
import urllib.request
import zipfile

data_dir = "./data"
# url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    # urllib.request.urlretrieve(url,target_path)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

target_path = os.path.join(data_dir,"archive.zip")

if not os.path.exists(target_path) :
    # Read file zip
    zip1 = zipfile.ZipFile(target_path)
    zip1.extractall(data_dir)
    zip1.close



from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil

# Function to Extract features from the images
def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    img_path = os.listdir(direc)
    for i in tqdm(img_path):
        fname=direc+'/'+i
        img=image.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features,img_name

#cluster = './CNN/DS/Dump/test2'
cluster = './SNN/DS/V4/10s/T5/plots/2'


img_features,img_name=image_feature(cluster)

print(img_features)
print(img_name)

#Creating Clusters
k = 2
clusters = KMeans(k)
clusters.fit(img_features)

image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = clusters.labels_
print(image_cluster) # 0 denotes cat and 1 denotes dog

# Made folder to seperate images
paths = []
for i in range(k):
	name = cluster + '/' +str(i)
	paths += [name]
	os.mkdir(name)
	for j in range(len(image_cluster)):
	    if image_cluster['clusterid'][j]==i:
	        shutil.copy(os.path.join(cluster, image_cluster['image'][j]), paths[i])
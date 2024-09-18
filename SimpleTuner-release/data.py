import base64
import ast 
from PIL import Image 
from io import BytesIO
import pandas as pd
import json

def decode(im_b64):
    image_bytes = base64.b64decode(ast.literal_eval(im_b64))
    img_files = BytesIO(image_bytes)
    return Image.open(img_files)


dataset = pd.concat([pd.read_csv("classic_fusion_train.csv"),
                     pd.read_csv("classic_fusion_val.csv")], ignore_index=True)


#for index, row in dataset.iterrows():
#    decode(dataset.iloc[index]["image"]).save(f'datasets/classic-fusion-hublot/{index+1}.jpg')
##    caption = {"caption" : dataset.iloc[index]["prompt"]}
#    text =  dataset.iloc[index]["prompt"]
#    with open(f"datasets/classic-fusion-hublot/{index+1}.txt", "w") as f:
#        #json.dump(caption, f)
#        f.write(text)

print(Image.open("datasets/classic-fusion-hublot/2.jpg").size)

left=0
top=187
right=906
bottom=1093
Image.open("datasets/classic-fusion-hublot/3.jpg").crop((left,top,right,bottom)).save('resized_image_3.jpg')
print(Image.open("resized_image_3.jpg").size)
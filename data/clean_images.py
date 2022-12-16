import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

def resize_img(final_size, im):
    size = im.size
    scaling_ratio = float(final_size/max(size))
    new_size = tuple([int(scaling_ratio * x) for x in size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_size[0])//2, (final_size-new_size[1])//2))
    return new_im

os.chdir("raw/images")
files = os.listdir()


imsizes = []
final_image_size = 512

for i in tqdm(range(len(files)), desc="reading images: "):
    try:
        im = Image.open(files[i])
        new_im = resize_img(final_image_size, im)
        new_im.save()
    except:
        pass

images = pd.DataFrame(imsizes)
print(images.describe)
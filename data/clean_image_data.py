from PIL import Image
from tqdm import tqdm
import os
import pandas as pd

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "data/raw/images/"
    save_path = "data/clean_images/"
    dirs = os.listdir(path)
    pd_dirs = pd.Series(dirs, name="id").apply(lambda x: x.split('.')[0])
    ref_imgs = pd.read_csv("data/raw/Images.csv")
    complete_imgs = pd.merge(pd_dirs, ref_imgs)[['id', 'product_id']]
    complete_imgs.to_csv("data/raw/Images_reduced.csv")

    print(complete_imgs.id)

    final_size = 48
    for item in tqdm(complete_imgs.id, desc="Images resized: "):
        try:
            im = Image.open(path + item + '.jpg')
            new_im = resize_image(final_size, im)
            new_im.save(f'{save_path}{item.split(".")[0]}_resized.jpg')
        except Image.UnidentifiedImageError:
            pass

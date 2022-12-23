from PIL import Image
from tqdm import tqdm
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "raw/images/"
    save_path = "data/clean_images/"
    dirs = os.listdir(path)
    final_size = 512
    for item in tqdm(dirs, desc="Images resized: "):
        try:
            im = Image.open(path + item)
            new_im = resize_image(final_size, im)
            new_im.save(f'{save_path}{item.split(".")[0]}_resized.jpg')
        except Image.UnidentifiedImageError:
            pass
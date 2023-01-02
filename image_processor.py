from torchvision.transforms import ToTensor
from PIL import Image

class ImageProcessor:

    def _transform_image(self, img):        
        if type(img) == str:
            img = Image.open(img)
        size = img.size
        final_size = 128
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        img = img.resize(new_image_size, Image.ANTIALIAS)
        new_img = Image.new("RGB", (final_size, final_size))
        new_img.paste(img, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        new_img = ToTensor()(new_img)
        new_img = new_img[None, :]
        return new_img

    def __call__(self, img):
        return self._transform_image(img)

import numpy as np
import squidpy as sq
from PIL import Image

def get_image(adata, to_uint8=True, scale=False):
    key = next(iter(adata.uns["spatial"].keys()))
    hires_image = adata.uns["spatial"][key]["images"]["hires"]

    image_container = sq.im.ImageContainer(hires_image)
    image_arr = image_container.data["image"].compute().values
    image_arr = np.squeeze(image_arr, axis=2)

    if to_uint8:
        image_arr = (255 * image_arr).clip(0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_arr)
    
    if scale:
        scalefactor = adata.uns["spatial"][key]["scalefactors"]["tissue_hires_scalef"]
        height, width = image_container.shape
        new_height = int(height / scalefactor)
        new_width = int (width / scalefactor)
        image = image.resize((new_width, new_height), resample=Image.BICUBIC)

    return image
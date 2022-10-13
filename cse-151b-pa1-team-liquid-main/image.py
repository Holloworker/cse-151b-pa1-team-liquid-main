from PIL import Image

def export_image(img_arr, name='test.tiff'):
    Image.fromarray(img_arr).save(name)

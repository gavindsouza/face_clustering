from PIL import Image 
from PIL.ExifTags import TAGS, GPSTAGS 
from pprint import pprint as print
import PIL.ExifTags


def get_exIF(image):
    exif = {
        TAGS[k]: v for k, v in image._getexif().items() if k in TAGS
    }
    return exif

def get_GPS(exif):
    gps_d = exif["GPSInfo"]
    gps = {
        GPSTAGS[k]: v for k, v in gps_d.items()  
    }
    return gps

image = Image.open("/home/gavin/Desktop/IMG_20180405_105159.jpg") 
# print(image._getexif())
print(get_GPS(get_exIF(image)))


image = Image.open("/home/gavin/Desktop/IMG_20180826_190300.jpg")
# print(image._getexif())
# print(get_exIF(image))
print(get_GPS(get_exIF(image)))

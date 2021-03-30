# imports - standard imports
from pprint import pprint as print

# imports - third party imports
from PIL import Image as pil_im 
from PIL.ExifTags import TAGS, GPSTAGS 
import numpy as np


class Image:
    def __init__(self, src: str):
        self.src = src
        self.image = pil_im.open(self.src)
        self.array = np.asarray(self.image)
        self.exif = {
            TAGS[k]: v for k, v in self.image._getexif().items() if k in TAGS
        }

    def __repr__(self):
        return "<%s.%s image at 0x%X>" % (self.__class__.__module__, self.__class__.__name__,id(self))

    @property
    def loc(self):
        return {
            GPSTAGS[k]: v for k, v in self.exif["GPSInfo"].items()  
        }

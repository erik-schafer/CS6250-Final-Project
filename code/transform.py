import os

from skimage import io, transform
import numpy as np


IMAGE_DIRECTORY = os.path.join("../data","images")
OUTPUT_DIRECTORY = os.path.join("../data", "image_transform")
TRANSFORM_SHAPE = (256,256)

def loadAndTransform():
    i = 0
    files = [f for f in os.listdir(IMAGE_DIRECTORY)]
    l = len(files)
    percent = int(l * .01)    
    print(f"1 percent {percent}")
    outFiles = os.listdir(OUTPUT_DIRECTORY)
    for i,f in enumerate(files):
        if f not in outFiles:         
            try:
                in_path = os.path.join(IMAGE_DIRECTORY, f)
                out_path = os.path.join(OUTPUT_DIRECTORY, f)

                im = io.imread(in_path)
                im = transform.resize(im, TRANSFORM_SHAPE)
                if im.shape[-1] == 3:
                    pass #it's already the correct shape
                elif im.shape[-1] == 4:
                    im = im[:,:,0:3]
                else:
                    im = np.stack([im]*3, axis=-1)
                io.imsave(out_path, im)
            except ValueError:
                print(f, "ValueError!")
        if i % percent == 0: print(i//percent, "% ")
        

def main():
    loadAndTransform()
    return

if __name__== "__main__":
    main()
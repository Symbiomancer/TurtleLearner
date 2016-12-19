import cv2
import os.path
import sys
import glob
import numpy as np
import math

def filter(x): 
    return 255 if  math.log1p(x) > 3.0 else 0


np.set_printoptions(threshold=np.inf)
f = np.vectorize(filter)
def generate_DVS_single(img1, img2):
    thresh_value = 3 #change as needed
    new_img = cv2.absdiff(img1, img2)
    """ 
    height, width = new_img.shape
    for i in range(0, height):
        for j in range(0, width):
            log_i = math.log1p(new_img[i][j])
            if log_i > float(thresh_value):
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0

    return new_img
    """
    return f(new_img)






def generate_DVS(folder, thresh_value):
    #num_images = len([name for name in os.listdir(folder) if os.path.isfile(name)])
    path, dirs, files = os.walk(folder + "/RGB_images").next()
    num_images = len(files)
    print num_images
    for a in xrange(num_images-1):
        print "reading image_"+ str(a) + ".png"
        img1 = cv2.imread(folder + "/RGB_images/" + "image_" + str(a) + ".png", 0)
        img2 = cv2.imread(folder + "/RGB_images/" + "image_" + str(a+1) + ".png", 0)
        cv2.imshow("img1",img1)
        cv2.imshow("img2",img2)

        new_img = cv2.absdiff(img1, img2)
        #cv2.imshow("diff", new_img)
        thresh_img = new_img.copy()
        print new_img.shape
        
        height, width = thresh_img.shape
        """
        for i in range(0,height):
            for j in range(0, width):
                log_i = math.log1p(thresh_img[i][j])
                if log_i > float(thresh_value):
                    thresh_img[i][j] = 255
                else:
                    thresh_img[i][j] = 0
        """
        #f = np.vectorize(filter)
        thresh_img = f(thresh_img)
        
        #print thresh_img
        #ret, thresh = cv2.threshold(new_img, 15, 255, cv2.THRESH_BINARY)
        #cv2.imshow("diff", new_img)
        #cv2.imshow("thresh", thresh)
        cv2.imwrite(folder + "/DVS_images/image_DVS_" + str(thresh_value) + "_" + str(a) + ".png", thresh_img)
    

#argument folder name, second threshold value
if __name__ == "__main__":
    generate_DVS(str(sys.argv[1]), int(sys.argv[2]))

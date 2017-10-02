
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np

from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(gray, window_size=64, orient=9, pix_per_cell=8, cell_per_block=2, svc=None):
    
    # (nyblocks, nxblocks, nblocks_per_window, nblocks_per_window, orient)
    
    # Define blocks and steps as above
    nxblocks_per_image = (gray.shape[1] // pix_per_cell) - cell_per_block + 1     # N - K + 1
    nyblocks_per_image = (gray.shape[0] // pix_per_cell) - cell_per_block + 1     # N - K + 1
    # (nyblocks, nxblocks, cell_per_block, cell_per_block, orient)
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window_size // pix_per_cell) - cell_per_block + 1
    print("nblocks_per_window", nblocks_per_window) # 7
    
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks_per_image - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks_per_image - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog_map = get_hog_features(gray, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    print(nyblocks_per_image, nxblocks_per_image, cell_per_block, cell_per_block, orient, hog_map.shape)    # (170, 853) (20, 105, 2, 2, 9)
    print(nysteps, nxsteps)         # (6, 49)

    print("==================================================================")
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            print(hog_map[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].shape)
            print(hog_map[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel().shape)
            
            hog_feat = hog_map[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            # start point
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            print(ytop, yb)
  
#             test_prediction = svc.predict(hog_map.reshape(1, -1))
#               
#                             if test_prediction == 1:
#                 xbox_left = np.int(xleft*scale)
#                 ytop_draw = np.int(ytop*scale)
#                 win_draw = np.int(window*scale)
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
#                   
#     return draw_img
    
ystart = 400
ystop = 656
img = mpimg.imread("test_images//test1.jpg")[:136, :136]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

out_img = find_cars(gray)

# plt.imshow(out_img)
#########################################################################################################################################









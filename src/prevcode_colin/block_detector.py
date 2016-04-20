import os
from pylab import *
# from matplotlib import cm
# from scipy.io import loadmat
# import scipy.ndimage as nd
import scipy
from scipy.ndimage import imread
# import pandas as pd
from skimage import feature
import cv2

def convert_depth_from_raw(im_depth):
    im_depth = np.load(dir_imgs+filename+"/"+xyz_filenames[0])['arr_0']
    im_depth.dtype = np.ubyte
    x = np.asarray((im_depth >> 16) & 255, dtype=np.int32)
    y = np.asarray((im_depth >> 8) & 255, dtype=np.int32)
    z = np.asarray(im_depth & 255, dtype=np.int32)
    im_out = np.dstack([x,y,z])
    print im_out.min(), im_out.max()
    return im_out

def cvtQuadColor(im):
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)*1.
    im_lab[:,:,1] -= 128
    im_lab[:,:,2] -= 128
    im_quad = np.empty((im_lab.shape[0], im_lab.shape[1], 4), np.float)
    # Green, red, blue, yellow
    im_quad[:,:,0] = (im_lab[:,:,1] < 0) * -im_lab[:,:,1]
    im_quad[:,:,1] = (im_lab[:,:,1] > 0) * im_lab[:,:,1]
    im_quad[:,:,2] = (im_lab[:,:,2] < 0) * -im_lab[:,:,2]
    im_quad[:,:,3] = (im_lab[:,:,2] > 0) * im_lab[:,:,2] 
    im_quad /= im_quad.sum(-1)[:,:,None]*1. + 1e-8
    im_quad = np.nan_to_num(im_quad)

    # for i in range(4):
    #     subplot(2,2,i+1)   
    #     imshow(im_quad[:,:,i])
    # show()

    return im_quad


def detect_green_mat(im):
    img = im[:,:,[2,1,0]]
    img_norm = img / (img.sum(-1)[:,:,None]+1e-8)
    hist_green = np.histogram(img[:,:,1], 256, (0,256))[0]




dir_imgs = "/Users/colin/Data/CogSci/raw/imgs/"
dir_out = "/Users/colin/Data/CogSci/viz/videos/"

rgb_suffix = "_rgb.jpg"
xyz_suffix = "_xyz.npz"

filename_num = 1

filename = os.listdir(dir_imgs)[0]
dir_trials = os.listdir(dir_imgs)
dir_trials = [f for f in dir_trials if f[0]!="_"]
# for filename in dir_trials:
if 1:
    filename = dir_trials[1]
    try:
    # if 1:
        print filename
        # filename = os.listdir(dir_imgs)[1]


        ims_filenames = os.listdir(dir_imgs+filename)
        rgb_filenames = np.sort([f.strip(rgb_suffix) for f in ims_filenames if f.find(".jpg")>0])
        xyz_filenames = np.sort([f.strip(xyz_suffix) for f in ims_filenames if f.find(".npz")>0])
        filenames = scipy.intersect1d(rgb_filenames, xyz_filenames)
        n_frames = len(filenames)

        im = imread(dir_imgs+filename+"/"+filenames[0]+rgb_suffix)
        depth_raw = np.load(dir_imgs+filename+"/"+filenames[0]+xyz_suffix)['arr_0']
        rez = im.shape

        # codec = cv2.VideoWriter_fourcc(*'X264')
        codec = cv2.cv.CV_FOURCC(*'X264')
        # vid = cv2.VideoWriter(dir_out+str(filename_num)+".mov", codec, 15.0, (rez[1], rez[0]))
        # vid = cv2.VideoWriter(dir_out+filename+".mov", codec, 15.0, (rez[1]*2, rez[0]))
        vid = cv2.VideoWriter(dir_out+filename+".mov", -1, 15.0, (rez[1]*2, rez[0]))

        # min_depth = depth_raw[depth_raw>0].min()
        max_depth = depth_raw.max() - 20
        min_depth = depth_raw.max() - 100

        mask_boundaries = np.zeros_like(depth_raw, np.bool)
        mask_boundaries[0:-100, 50:-100] = 1

        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255
        params.filterByInertia = False
        params.filterByConvexity = False
        params.minArea = 500
        params.maxArea = 3000
        blob_det = cv2.SimpleBlobDetector(params)

        print "frames:", n_frames

        # Get clips
        for i in range(0, n_frames, 10):

            # Load image
            im_rgb = imread(dir_imgs+filename+"/"+filenames[i]+rgb_suffix)[:,:,[2,1,0]]
            # im_rgb = imread(dir_imgs+filename+"/"+filenames[i]+rgb_suffix)
            # im_rgb = cv2.GaussianBlur(im_rgb, (15,15), 5., 5.)
            im_rgb = cv2.medianBlur(im_rgb, 11)


            depth = np.load(dir_imgs+filename+"/"+filenames[i]+xyz_suffix)['arr_0']

            mask = (depth > min_depth)*(depth < max_depth)
            mask *= mask_boundaries
            depth[depth>0] = (depth[depth>0]-min_depth)/(max_depth-min_depth)
            depth[depth<0] = 0
            depth *= mask#[:,:,None]

            # im = np.hstack([im, depth.astype(np.uint8)])
            # im = np.hstack([im*mask[:,:,None], depth.astype(np.uint8)])

            im_norm = im_rgb / (im_rgb.sum(-1)[:,:,None]+1e-8)
            im_quad = cvtQuadColor(im_rgb)

            im = (im_norm*255).astype(np.uint8)

            # edges2 = feature.canny(im[:,:,1], sigma=2)
            # bbox = cv2.boundingRect(np.array(np.nonzero(edges2)).T)


            im *= mask[:,:,None]
            im = np.ascontiguousarray(im)
            for i in range(4):
                tmp = (im_quad[:,:,i]*mask*1*255).astype(np.uint8)
                blobs = blob_det.detect(tmp)
                # print "Blobs", len(blobs)
                for bb in blobs:
                    pt = (int(bb.pt[0]), int(bb.pt[1]))
                    cv2.circle(im, pt, 3, (255,255,255), -1)

            # cv2.rectangle(im, (bbox[0], bbox[1])
            # im = nd.maximum_filter(im[:,:,1], 5)
            # contours = cv2.findContours(cv2.Canny(im, 10, 100), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            im_comb = np.empty([480, 640*2, 3], dtype=np.uint8)
            im_comb[:,:640,:] = im_rgb
            im_comb[:,640:,:] = im

            vid.write(im_comb)
            # cv2.imshow("img", im_norm*mask[:,:,None])
            cv2.imshow("img", im)
            # cv2.imshow("img", pred/pred.max())
            cv2.imshow("depth", im_rgb)

            ret = cv2.waitKey(10)
            if ret >= 0:
                break





        vid.release()
    except:
        print "Error outputing", filename_num


            # output_path = "/home/colin/Data/Surgical/viz/alexnet_viz/"+feat+"/"
            # if not os.path.exists(output_path):
            #     os.mkdir(output_path)
            # savefig(output_path+filename_num+"_"+feat+".jpg")

            # from sklearn.mixture import DPGMM, GMM
            # clst = GMM(4)
            # tmp = im_quad[mask].reshape([-1,4])
            # if tmp.shape[0] == 0:
            #     continue
            # clst.fit(tmp)
            # p = clst.predict(tmp)
            # pred = np.zeros_like(mask, np.float)
            # pred[mask] = p+1
import cv2
import glob
from scipy.spatial.distance import cosine
import numpy as np
import itertools
import shutil
import os 
import sys
from joblib import Parallel, delayed

## include sift and flann modules
sift = cv2.SIFT()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

#load image and resize it
def load_img(fl, resize=None):
	img = cv2.imread(fl)
	if resize:
		img=cv2.resize(img, (256,256))
	return img

## load all the images from the folder and calculate color histogram for it
def init(folder):

	Images=[]	
	for fl in glob.glob('%s/*.bmp'%folder):
		img=load_img(fl, True)
		#kp, desc = sift.detectAndCompute(img,None)
		desc = 0.0
		hist_ref= cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
		hist_ref= cv2.normalize(hist_ref).flatten()
		Images.append([img, hist_ref, 0.0, fl])
		
	return Images

## sift matching function
def sift_match(ref, img):
	
	matches = flann.knnMatch(img,ref,k=2)

	count = 0.0
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
			count+=1

	return count

## color histogram matching
def match_col_hist(ref, img):

    return -cv2.compareHist(ref, img, cv2.cv.CV_COMP_BHATTACHARYYA)

## match template and candidate image
def match(ref, matchImage, match_func):

	matches=[]	
	for index, refImage in enumerate(ref):
		for index, fl in enumerate(matchImage):
			matchImage[index][2] += match_func(refImage[1], fl[1])/len(ref)

	return sorted(matchImage, key=lambda x: -x[2])

## reset descriptor and distance value
def descriptor(img):

	for index, fl in enumerate(img):
		kp, desc = sift.detectAndCompute(fl[0],None)
		img[index][1] = desc
		img[index][2] = 0.0

## plot image 
def show_img(mat):
	cv2.imshow('disp', mat)

def doParallel(fl, prefix):
	reference = init('ref/')
	matchImage = init(fl + '/')

	## call matching color histogram for template and candidte image
	cartoon_matches = match(reference, matchImage, match_col_hist)
	cartoon_matches = [c for c in cartoon_matches if c[2]>-0.64]

	descriptor(reference)
	descriptor(cartoon_matches)

	cartoon_matches = match(reference, cartoon_matches, sift_match)
	cartoon_matches = [c for c in cartoon_matches if c[2] > 0.26]

	## iterate over the best candidate image after match
	for idx, _match in enumerate(cartoon_matches):
		img, desc, dist, name = _match
		print idx, dist, name 
		dest = os.path.join(output, prefix + str(idx) +'.bmp')
		shutil.copyfile(name, dest)

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(dir_path, sys.argv[1])
output = os.path.join(dir_path, sys.argv[2])

#Parallel(n_jobs=1, verbose=5)(delayed(doParallel)(fl, str(idx)) for idx,fl in enumerate(glob.glob('%s/*'%folder)))	
for idx,fl in enumerate(glob.glob('%s/*'%folder)):
	doParallel(fl, str(idx)) 
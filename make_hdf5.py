# script to preprocess the image and save in hdf5.
import cPickle as pickle
import h5py
import numpy as np
import skimage.io
import skimage.transform
import glob
from sklearn.utils import shuffle as sk_shuffle
from joblib import Parallel, delayed
import os.path
from sklearn.cross_validation import train_test_split

#------------------------------util------------------------------#
def read_lines(flname):
	with open(flname,'r') as f:
		return f.read().strip().split('\n')

def parallelize(func, itemlist, params={}, verbose=1):
	return Parallel(n_jobs=20, verbose=verbose)(delayed(func)(item, params) for item in itemlist)

def read_img(fl, params): #skimage gives you an RGB, cv2 gives BGR
	img=skimage.io.imread(fl)
	#img=skimage.transform.resize(img, (128,128))

	try:
		img=np.rollaxis(img,2) #(3,256,256)
		img=img[::-1,:,:] #convert RGB to BGR
	except:
		#shutil.copyfile(fl, 'failed/'+fl)
		print 'fail', fl
		return None

	if params==1:
		img=img[0,np.newaxis] #keep only channel 0


	#if >80% of the image is black, ditch.
	ch1=img[0]
	if 1.0*np.sum(ch1[ch1<10])/(ch1.shape[0]*ch1.shape[1])>0.8:
		return None

	return img

#----------------------------------------------------------------#

def make_hdf5(filenames, out_file, shuffle=True):


	color_files=['raw/frames/%s'%f for f in filenames]
	sketch_files=['raw/sketch/%s'%f for f in filenames]

	color_imgs=parallelize(read_img, color_files, verbose=10)
	sketch_imgs=parallelize(read_img, sketch_files, verbose=10, params=1)

	img_data=[(c,s) for c,s in zip(color_imgs, sketch_imgs) if (c is not None and s is not None)]

	img_data=sk_shuffle(img_data)
	color_imgs, sketch_imgs=zip(*img_data)
	color_imgs= np.asanyarray(color_imgs)
	grey_imgs=np.mean(color_imgs, 1, keepdims=True)
	sketch_imgs= np.asanyarray(grey_imgs)
	
	img_mean=np.mean(color_imgs, 0)

	with h5py.File(out_file, 'w') as hf:
		hf.create_dataset('col_sketch_data', data=color_imgs)
		hf.create_dataset('bw_sketch_data', data=sketch_imgs)
		hf.create_dataset('col_reference_data', data=color_imgs)		
		hf.create_dataset('img_mean', data=img_mean)

#--------------------------------------------------------------------#

img_files=[fl.split('/')[-1] for fl in glob.glob('raw/frames/*.png')]

train_files, val_files, _, _ = train_test_split(img_files, [0]*len(img_files), test_size=0.2, random_state=42)
make_hdf5(train_files, 'hdf5/train_data.h5', True)
make_hdf5(val_files, 'hdf5/val_data.h5', False)








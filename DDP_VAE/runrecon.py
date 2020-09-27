# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

import numpy as np
import os
import pickle
import vaerecon
import sys   #tiaoshi shiyong
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from US_pattern import US_pattern
import time
import h5py as h5
import cv2
import glob
# to tell tensorflow which GPU to use
#-----------------------------------------------------
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'; 

from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())

#print( os.environ['SGE_GPU'])

#print( '0' ) #wsy add


#which VAE model to use
#-----------------------------------------------------
ndims=28 # patch size
lat_dim=60 # latent space size

from scipy.io import loadmat,savemat
#load the test image and make uniform coil maps
#-----------------------------------------------------
#print(orim)
#print(orim.shape)
#print(orim.dtype)
#orim = np.array(orim,dtype=np.complex128)
#orim = np.rot90(orim)
print("--------------------------------------")
#print(np.max(orim),np.min(orim))
#print(orim.dtype)
#print('(orim.dtype)',orim.shape)
# or use pickle.load(open('./orim','rb'))
#sensmaps=np.ones_like(orim)
#sensmaps=sensmaps[:,:,np.newaxis]   # return a array that all 1 and the form is the same as orim
#print(sensmaps)
# load the US pattern or make a new one
#-----------------------------------------------------
uspat = loadmat('./SIAT/mask_radial_030.mat')['mask_radial_030']
uspat = np.fft.fftshift(uspat)
#-----------------------------------------------------
test_path = glob.glob('./SIAT_test_image31/*.mat')
test_path.sort()
method = 'radial30'
result_all = np.zeros([32,3])
for i,path in enumerate(test_path[:]):
    #i += 3
    orim = abs(loadmat(path)['Img'])           ##real
    #orim = loadmat(path)['Img']               ######complex
    #orim = abs(orim)/np.max(np.abs(orim))
    import matplotlib.pyplot as plt
    '''
    plt.figure(1)
    plt.imshow(uspat,cmap='gray')
    plt.show()

    plt.figure(1)
    plt.imshow(orim,cmap='gray')
    plt.show()
    '''
    #assert False

    usksp = (np.fft.fft2(orim)*uspat)#/np.percentile(np.abs(np.fft.ifft2(np.fft.fft2(orim)*uspat)).flatten()  ,99)
    #usksp = UFT(orim,uspat)/np.percentile( np.abs(tUFT(UFT(orim,uspat),uspat).flatten())  ,99) #notice the usksp has zeros at positions which are not sampled.
    zerofilled = np.fft.ifft2(usksp)
    print('usksp shape',usksp.shape)
    print('usksp dtype',usksp.dtype)
    print(np.max(usksp.real))
    print(np.max(usksp.imag))
    print(np.max(orim.real))
    print(np.min(orim.imag))
    print(np.max(zerofilled.real))
    print(np.min(zerofilled.real))
    #assert False
    #plt.figure(6)
    #plt.imshow(abs(zerofilled).reshape(256,-1), cmap=cm.gray)  #232 252
    #plt.show()
    # reconstruction parameters
    #-----------------------------------------------------
    regtype='reg2' # which type of regularization/projection to use for the phase image. 'TV' also works well...
    reg=0.1# strength of this phase regularization. 0 means no regularization is applied
    regiter = 10 # how many iterations for the phase regularization

    num_pocs_iter = 50 # number of total POCS iterations
    dcprojiter=1 # there will be a data consistency projection every 'dcprojiter'steps

    num_iter = 300 #num_pocs_iter*dcprojiter+2 # how many total iterations to run the reconstruction. 

    # note: this setting corresponds to 302/10 = 30 POCS iterations,
    #since you do a data consistency projection every ten steps.
    #the extra 2 are necessary to make sure the for loop runs until the last data
    #consistency projection.
    #notice you need to take num_iter some multiple of dcprojiter + 2, so that the data consistency
    #projection runs as the last step.

    parfact = 30 # a factor for parallel computing for speeding up computations,
    #i.e. doing operations in parallel for the patches, but upper bounded by memory 

    # run the recon!
    print('current image process is',i,'png')
    rec_ddp = vaerecon.vaerecon(usksp,uspat,orim,i,result_all,method, dcprojiter=dcprojiter, lat_dim=lat_dim, patchsize=ndims ,parfact=parfact, num_iter=num_iter, regiter=regiter, reglmb=reg, regtype=regtype)
    #rec_ddp = np.reshape(rec_ddp,[orim.shape[0], orim.shape[1], -1])
    #pickle.dump(rec_ddp, open('./rec_{}'.format(i) ,'wb')   )
pickle.dump(np.abs(tFT(usksp)), open('./zerofilled' ,'wb')   )
               
'''
# the reconstructed image is the image after the last data consistency projection
rec = rec_ddp[:,:,-1] # i.e. the 301th image

rec_abs = np.abs(rec)
rec_phase = np.angle(rec)


# calculate RMSE while making sure the images are scaled similarly:
rmse = calc_rmse(orim , rec_abs/np.linalg.norm(rec_abs)*np.linalg.norm(orim) )   
print(rmse)      '''

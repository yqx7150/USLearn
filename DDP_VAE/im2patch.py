# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

import numpy as np
import time
import h5py as h5
import matplotlib.pyplot as plt
class Patcher:
     # Class that creates patches of images
     
     #you need to make sure that the image size is a multiple of patchsize
     #but you can have a step size, where you do not get full patches.
     #您需要确保图像大小是补丁大小的倍数,但是你可以有一个步骤大小，在那里你不会得到完整的补丁
     #====================================================================
     def __init__(self, imsize, patchsize, step, nopartials, contatedges):
          self.patchsize = patchsize
          self.step = step
          self.imsize=imsize
          self.imsizeorig=np.array(imsize)
          self.nopartials = nopartials
          self.contatedges = contatedges
          
          self.diviim =[]
          self.genpatchsizes=[]
          self.noOfPatches=0
          
          
          
          #if you want to be able to use patchsizes not dividor of image size
          if self.contatedges:
               if (self.imsize == (np.ceil(self.imsizeorig/self.patchsize)*self.patchsize).astype(int)).all():
                    self.contatedges=False
               else:
                    self.imsize = (np.ceil(self.imsizeorig/self.patchsize)*self.patchsize).astype(int)
               
          self.getDivImage()
          
          
          
     def im2patches(self,img):
          
          #pad images with srap to make the image size multiple of patchsize
          if self.contatedges:
               sd=self.imsize - self.imsizeorig
               img = np.pad(img,[ (0,sd[0]), (0,sd[1])  ], mode='wrap'  )
          
          ptchs=[]
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    
                    ptc = img[ix:ix+self.patchsize, iy:iy+self.patchsize]
                    
                    if ((ptc.shape[0] != self.patchsize) or (ptc.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else:  
                         ptchs.append(ptc)
               
          return ptchs
          
          
          
     def patches2im(self,patches, combsq=False):
          
          if len( self.diviim):
               pass
          else:
               self.getDivImage()
              
          if self.contatedges:     
               tmp=np.zeros(self.imsize, dtype=np.complex128)
          else:
               tmp=np.zeros(self.imsizeorig, dtype=np.complex128)
               
          ctr=0
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    
                    tt=tmp[ix:ix+self.patchsize, iy:iy+self.patchsize]
                    
                    
                    if ((tt.shape[0] != self.patchsize) or (tt.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else:
                         tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] = tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] + patches[ctr]
                         ctr=ctr+1
          
          if not combsq:
               tmp=tmp/self.diviim
          else:
               tmp=tmp/np.square(self.diviim)
               
          tmp=tmp[0:self.imsizeorig[0], 0:self.imsizeorig[1]]
          
          return tmp
          
          
     def getDivImage(self):
          
          if self.contatedges:
               tmp=np.zeros(self.imsize)
          else:
               tmp=np.zeros(self.imsizeorig)
               
          gensizes=[]
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    tt=tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] 
                    
                    if ((tt.shape[0] != self.patchsize) or (tt.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else: 
                         gensizes.append(tt.shape)#keep this to check patch sizes later
                         tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] = tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] + 1
          
          if (tmp==0).any():
               print("KCT-WARNING: the selected patching scheme does not allow covering of all the image! Some pixels are not in any of the patches.")
               
          tmp[np.where(tmp==0)]=1 #do as if full coverage were provided anyways...
             
          self.diviim = tmp
          self.genpatchsizes = gensizes
          self.noOfPatches = len(gensizes)


#%% This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

SEED=1001
seed=1
np.random.seed(seed=seed)

def getData(trnTst='testing',num=100,sigma=.01):
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data
    print('Reading the data. Please wait...')
    filename='dataset.hdf5' #set the correct path here
    tic()
    with h5.File(filename) as f:
        if trnTst=='training':
            org,csm,mask=f['trnOrg'][:],f['trnCsm'][:],f['trnMask'][:]
        else:
            org,csm,mask=f['tstOrg'][num],f['tstCsm'][num],f['tstMask'][num]
            na=np.newaxis
            org,csm,mask=org[na],csm[na],mask[na]
    toc()
    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    tic()
    #%%  add dowmsample function
    toc()
    print('Successfully undersampled data!')
    #if trnTst=='testing':
    #    atb=c2r(atb)
    return org,csm,mask

orgdata,_,_ = getData(trnTst='training')
print(orgdata.shape)    
'''plt.figure(1)
plt.imshow(np.abs(orgdata[1,:,:].reshape(256,232)),cmap='gray')
plt.show()'''
Ptchr=Patcher(imsize=[256,232],patchsize=28,step=int(28/2), nopartials=True, contatedges=True)   
nopatches=len(Ptchr.genpatchsizes)
sumptchs = []
for i in range(2):
     ptchs = Ptchr.im2patches(np.reshape(orgdata[i,:,:].reshape(orgdata.shape[1],orgdata.shape[2]), [256,232]))
     #ptchs=np.array(ptchs)
     sumptchs = sumptchs + ptchs
sumptchs=np.array(sumptchs)
print(sumptchs.shape)
sumptchs = sumptchs.reshape(646,784)
sumptchs = sumptchs.reshape(646,28,28)
#print(ptchs.shape)
img = Ptchr.patches2im(ptchs)
img=np.array(img)
print(img.shape)
plt.figure(2)
plt.imshow(np.abs(img),cmap='gray')
plt.show()
print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")                   
          
               
               
                              
               


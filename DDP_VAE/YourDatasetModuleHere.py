# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich


import numpy as np
from scipy import misc   # tiaoshishiyong
import h5py as h5
import time
from Patcher import Patcher
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%% This provide functionality similar to matlab's tic() and toc()
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

def add_Usp_nous(img):
    #print("11111111111111111",img.shape)
    #img = np.rot90(img)
    #print(img.shape)
    out_img = abs(img/np.max(np.abs(img)))
    #print('np.max(out_img),np.min(out_img)',np.max(out_img),np.min(out_img))
    #assert False
    return out_img
def getData(trnTst='training',num=100,sigma=.01):
    import glob
    from scipy.io import loadmat
    #paths = glob.glob('../SIATdata_500mat_64_dataaug_6ch/real_imag/*.mat')
    #datas = np.zeros([8000,64,64,1])
    if trnTst == 'training':
        paths = glob.glob('./matdata/*.mat')
        datas = np.zeros([500,256,256,1],dtype=np.complex128)
        for i,each_path in enumerate(paths):
            data = loadmat(each_path)["Img"]
            #data = data[:,:,0]+data[:,:,1]*1j
            #print(data.shape)
            datas[i,:,:,0] = data
    else:
        paths = glob.glob('./test_mat/*.mat')
        datas = np.zeros([4,256,256,1],dtype=np.complex128)
        for i,each_path in enumerate(paths):
            data = loadmat(each_path)["Img"]
            #data = data[:,:,0]+data[:,:,1]*1j
            #print(data.shape)
            datas[i,:,:,0] = data
    return datas

class YourDatasetModuleHere(object):

    def __init__(self):
        self.trainptchs = []
        self.traincount = 1
        self.testptchs = []
        self.testcount = 1
        sumptraintchs = []
        sumptesttchs = []
        
        orgtraindata = getData(trnTst='training')
        print(orgtraindata.shape)
        orgtestdata= getData(trnTst='testing')
        
        print('orgtraindata',orgtraindata.dtype,type(orgtraindata))
        
        Ptchr=Patcher(imsize=[orgtraindata.shape[1],orgtraindata.shape[2]],patchsize=28,step=int(28/2), nopartials=True, contatedges=True)   
        
        #Ptchr1=Patcher(imsize=[orgtestdata.shape[1],orgtestdata.shape[2]],patchsize=28,step=int(28/2), nopartials=True, contatedges=True)
        for i in range(500):
            ptchs_1 = Ptchr.im2patches(add_Usp_nous(orgtraindata[i,:,:,:]).reshape([256,256]))
            sumptraintchs += ptchs_1
            #orgtraindata[i,:,:,0] = np.reshape(add_Usp_nous(orgtraindata[i,:,:,:]).reshape([64,64]), [64,64])
        for i in range(4):
            ptchs_1 = Ptchr.im2patches(add_Usp_nous(orgtestdata[i,:,:,:]).reshape([256,256]))
            sumptesttchs += ptchs_1
        sumptraintchs=np.array(sumptraintchs)
        sumptraintchs = sumptraintchs.reshape(-1,784)
        print(sumptraintchs.shape)
        self.trainptchs = sumptraintchs
        
        sumptesttchs=np.array(sumptesttchs)
        sumptesttchs = sumptesttchs.reshape(-1,784)
        self.testptchs = sumptesttchs
        
    def get_train_batch(self,batch_size=50,flag=0):
        if flag == 1:
            self.traincount = 1
        train_batch = self.trainptchs[batch_size*self.traincount-batch_size:batch_size*self.traincount,:]

        self.traincount += 1
        #print(self.traincount)
        return train_batch
        
        
    def get_test_batch(self,batch_size=50):
        test_batch = self.testptchs[batch_size*self.testcount-batch_size:batch_size*self.testcount,:]

        self.testcount += 1
        #print(self.traincount)
        return test_batch
     
     
if __name__=="__main__":
    DS = YourDatasetModuleHere()
    #batch = DS.get_train_batch()
    #print(batch.shape)
    '''orgtraindata = getData(trnTst='training')
    orgtestdata = getData(trnTst='testing')
    #orgtraindata1 = getTestData(trnTst='training')
    orgtestdata1 = getTestData(trnTst='testing')
    plt.imshow(abs(orgtestdata1).reshape(256,232),cmap=cm.gray)
    plt.show()
    #print("xxxxxx",orgtraindata1.shape)
    print("xxxxxx",orgtestdata1.shape)
    print("xxxxxx",orgtraindata.dtype)
    orgtestdata = orgtestdata.reshape(164,256,232)
    print("xxxxxx1",orgtestdata.reshape(164,256,232).shape)
    plt.figure(1)
    plt.ion()
    for i in range(164):
        print(i)
        print(orgtestdata[i,:,:].shape)
        plt.imshow(abs(orgtestdata[i+30,:,:]).reshape(256,232),cmap=cm.gray)
        plt.pause(0.1)'''
        

# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

from __future__ import division
from __future__ import print_function
import numpy as np

from Patcher import Patcher
from definevae import definevae

import scipy.io


import time
from scipy.io import loadmat,savemat
import os
import subprocess
import sys
from compare_hfen import compare_hfen
from skimage.measure import compare_psnr,compare_ssim
import matplotlib.pyplot as plt
import cv2
savepath = './radial30_rec'

def vaerecon(us_ksp_r2,uspat,orim,i,result_all,method, dcprojiter, onlydciter=0, lat_dim=60, patchsize=28, contRec='', lowresmodel=False, parfact=10, num_iter=1, regiter=15, reglmb=0.05, regtype='TV'):
     
     def write_Data(result_all):
        with open(os.path.join(savepath,"psnr_"+method+".txt"),"w+") as f:
            #print(len(result_all))
            for i in range(len(result_all)):
                f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
                "    SSIM : " + str(result_all[i][1]) + "    HFEN : " + str(result_all[i][2]))
                f.write('\n')
     print('KCT-INFO: contRec is ' + contRec)
     print('KCT-INFO: parfact is ' + str(parfact) )
     #print("............",us_ksp_r2.shape)
     #print(sensmaps.shape)
     # set parameters
     #==============================================================================
     np.random.seed(seed=1)
     
     imsizer=us_ksp_r2.shape[0] #252#256#252
     imrizec=us_ksp_r2.shape[1] #308#256#308
     #print(imsizer,imrizec)
     nsampl=50
          
     
     #make a network and a patcher to use later
     #==============================================================================
     x_rec, x_inp, funop, grd0, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = definevae(lat_dim=lat_dim, patchsize=patchsize, batchsize=parfact*nsampl)

     Ptchr=Patcher(imsize=[imsizer,imrizec],patchsize=patchsize,step=int(patchsize/2), nopartials=True, contatedges=True)   
     nopatches=len(Ptchr.genpatchsizes)
     print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")
     

     def dconst(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          
          return np.linalg.norm(np.fft.fft2(us)*uspat - data ) **2
          
     def dconst_grad(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          return 2*np.fft.ifft2(uspat*(np.fft.fft2(us)*uspat - data))
     def prior(us):
          #inp: [parfact,ps*ps]
          #out: parfact
          
          us=np.abs(us)
          funeval = funop.eval(feed_dict={x_rec: np.tile(us,(nsampl,1)) }) # 
          funeval=np.array(np.split(funeval,nsampl,axis=0))# [nsampl x parfact x 1]
          return np.mean(funeval,axis=0).astype(np.float64)
     
     def prior_grad(us):
          #inp: [parfact, ps*ps]
          #out: [parfact, ps*ps]
          #print('prior_grad',us.dtype,np.max(us.imag))
          usc=us.copy()
          usabs=np.abs(us)
          #print('prior_grad usabs',usabs.dtype,np.max(usabs.imag))
          #print('prior_grad usc',usc.dtype,np.max(usc.imag))
          #print('usc == us ? :',usc == us)
          grd0eval = grd0.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          #grd0eval: [500x784]
          grd0eval=np.array(np.split(grd0eval,nsampl,axis=0))# [nsampl x parfact x 784]
          grd0m=np.mean(grd0eval,axis=0) #[parfact,784]
          
          #print('grd0eval.shape,grd0m.shape',grd0eval.shape,grd0m.shape)
          #print(grd0m)
          
          grd0m = usc/np.abs(usc)*grd0m
          
          #print('prior_grad grd0m',grd0m.dtype,np.max(grd0m.imag))       

          return grd0m  # .astype(np.float64)
     
     def prior_grad_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: [np, ps, ps] 
          #takes set of patches as input and returns a set of their grad.s 
          #both grads are in the positive direction
          
          shape_orig=ptchs.shape
          
          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1] )
          
          grds=np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact), np.prod(ptchs.shape[1:])], dtype=np.complex128)
          
          extraind=int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs=np.pad(ptchs,( (0,extraind),(0,0)  ), mode='edge' )
          
          #print('prior_grad_patches',ptchs.dtype,np.max(ptchs.imag))
          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               grds[parfact*ix:parfact*ix+parfact,:]=prior_grad(ptchs[parfact*ix:parfact*ix+parfact,:]) 
               
                  
          grds=grds[0:shape_orig[0],:]

          print('np.reshape(grds, shape_orig).shape',np.reshape(grds, shape_orig).shape)
          return np.reshape(grds, shape_orig)
          #np.ceil   shang qu zheng

     def prior_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: 1
          
          fvls=np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact) ])
          
          extraind=int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs=np.pad(ptchs,[ (0,extraind),(0,0), (0,0)  ],mode='edge' )
          
          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               fvls[parfact*ix:parfact*ix+parfact] = prior(np.reshape(ptchs[parfact*ix:parfact*ix+parfact,:,:],[parfact,-1]) )
               
          fvls=fvls[0:ptchs.shape[0]]
               
          return np.mean(fvls)
          
     def full_gradient(image):
          #inp: [nx*nx, 1]
          #out: [nx, ny], [nx, ny]
          
          #returns both gradients in the respective positive direction.
          #i.e. must 
          
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs=np.array(ptchs)
          #print('ptchs dtype',ptchs.dtype,np.max(ptchs))
          
          grd_prior = prior_grad_patches(ptchs)
          grd_prior = (-1)* Ptchr.patches2im(grd_prior)
          
          grd_dconst = dconst_grad(np.reshape(image, [imsizer,imrizec]))
          
          return grd_prior + grd_dconst, grd_prior , grd_dconst
     def full_funceval(image):
          #inp: [nx*nx, 1]
          #out: [1], [1], [1]
          
          tmpimg = np.reshape(image, [imsizer,imrizec])
          
          dc = dconst(tmpimg)
     
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs=np.array(ptchs)
          
          priorval = (-1)*prior_patches(np.abs(ptchs))
          
          
          return priorval + dc, priorval, dc    
     
     #define the phase regularization functions
     #==============================================================================
     
     def tv_proj(phs,mu=0.125,lmb=2,IT=225):
          # Total variation based projection
          
          phs = fb_tv_proj(phs,mu=mu,lmb=lmb,IT=IT)
          
          return phs
     
     def fgrad(im):
          # gradient operation with 1st order finite differences
          imr_x = np.roll(im,shift=-1,axis=0)
          imr_y = np.roll(im,shift=-1,axis=1)
          grd_x = imr_x - im
          grd_y = imr_y - im
          
          return np.array((grd_x, grd_y))
     
     def fdivg(im):
          # divergence operator with 1st order finite differences
          imr_x = np.roll(np.squeeze(im[0,:,:]),shift=1,axis=0)
          imr_y = np.roll(np.squeeze(im[1,:,:]),shift=1,axis=1)
          grd_x = np.squeeze(im[0,:,:]) - imr_x
          grd_y = np.squeeze(im[1,:,:]) - imr_y
          
          return grd_x + grd_y
     
     def f_st(u,lmb):
          # soft thresholding
          
          uabs = np.squeeze(np.sqrt(np.sum(u*np.conjugate(u),axis=0)))
          
          tmp=1-lmb/uabs
          tmp[np.abs(tmp)<0]=0
             
          uu = u*np.tile(tmp[np.newaxis,:,:],[u.shape[0],1,1])
          
          return uu
     
     
     def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
          
          sz = im.shape
          us=np.zeros((2,sz[0],sz[1],IT))
          us[:,:,:,0] = u0
          
          for it in range(IT-1):
               
               #grad descent step:
               tmp1 = im - fdivg(us[:,:,:,it])
               tmp2 = mu*fgrad(tmp1)
               
               tmp3 = us[:,:,:,it] - tmp2
                 
               #thresholding step:
               us[:,:,:,it+1] = tmp3 - f_st(tmp3, lmb=lmb)     
               
          #endfor     

          return im - fdivg(us[:,:,:,IT-1])
     
   
     
     def reg2_proj(usph, niter=100, alpha=0.05):
          #A smoothness based based projection. Regularization method 2 from
          #"Separate Magnitude and Phase Regularization via Compressed Sensing",  Feng Zhao et al, IEEE TMI, 2012
          
          usph=usph+np.pi*3/2
          
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          for ix in range(niter-1):
              ims[:,:,ix+1] = ims[:,:,ix]-2*alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
          
          return ims[:,:,-1]-np.pi*3/2
     
     #make the data
     #===============================
     
     uspat=np.abs(us_ksp_r2)>0
     uspat=uspat[:,:]
     data=us_ksp_r2
        
          
     import pickle

     #make the functions for POCS
     #===================================== 
     #number of iterations
     numiter=num_iter
     
     # if you want to do an affine data consistency projection
     multip = 0 # 0 means simply replacing the measured values
     
     # step size for the prior iterations
     alphas=np.logspace(-4,-4,numiter)                                                   
     #print('step size for the prior iterations',alphas)
     #assert False
     
     
     def feval(im):
          return full_funceval(im)
          
     def geval(im):
          t1, t2, t3 = full_gradient(im)
          return np.reshape(t1,[-1]), np.reshape(t2,[-1]), np.reshape(t3,[-1])
     
     # initialize data with the zero-filled image
     recs=np.zeros((imsizer*imrizec,numiter), dtype=complex) 
     #recs[:,0] = tUFT(data, uspat).flatten().copy() 
     
     recs[:,0] = np.fft.ifft2(data*uspat).flatten().copy() 
     recs[:,0] = abs(recs[:,0])
     print('Zerosfilled Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,0],[imsizer,imrizec])),255*np.abs(orim),data_range=255))
     #plt.figure(444444444)
     #plt.imshow(np.abs(np.reshape(recs[:,0],[imsizer,imrizec])),cmap='gray')   
     #plt.show()
     # if you want to instead continue reconstruction from an existing image
     print(' KCT-INFO: contRec is ' + contRec)
     if contRec != '':
          print('KCT-INFO: reading from a previous file '+contRec)
          rr=pickle.load(open(contRec,'rb'))
          recs[:,0]=rr[:,-1]
          print('KCT-INFO: initialized to the previous recon: ' + contRec)
          

     import time 
     # the itertaion loop
     max_psnr = 0
     max_ssim = 0
     min_hfen = 100
     for it in range(numiter-1):
          start = time.time()
          # get the step size for the Piteration
          alpha=alphas[it]
          
          # if you want to do some data consistency iterations before starting the prior projections
          # this can be helpful when doing recon with multiple coils, e.g. you can do pure SENSE in the beginning...
          # or only do phase projections in the beginning.
          if it > onlydciter:
               
                # get the gradients for the prior projection and the likelihood values
                #ftot, f_prior, f_dc = feval(recs[:,it])
                print('recs dtype Input ',recs.dtype)
                gtot, g_prior, g_dc = geval(recs[:,it])
               
#                print("it no: " + str(it) + " f_tot= " + str(ftot) + " f_prior= " + str(-f_prior) + " f_dc (x1e6)= " + str(f_dc/1e6) + " |g_prior|= " + str(np.linalg.norm(g_prior)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )
                #print("it no: {0}, f_tot= {1:.2f},f_prior= {2:.2f}, f_dc (x1e6)= {3:.2f}, |g_prior|= {4:.2f}, |g_dc|= {5:.2f}".format(it,ftot,f_prior,f_dc/1e6,np.linalg.norm(g_prior),np.linalg.norm(g_dc)))
                          
                # update the image with the prior gradient
                recs[:,it+1] = recs[:,it] - alpha*g_prior

                print('current Phase prePsnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec])),255*np.abs(orim),data_range=255))
                print('current Phase Real Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).real),255*np.abs(orim.real),data_range=255))
                print('current Phase Imag Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).imag),255*np.abs(orim.imag),data_range=255))
                
                #plt.figure(1111)
                #plt.imshow(np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec])),cmap='gray')  
                #plt.show()
                #print("............................................")
                #print(recs[:,it+1])
                # seperate the magnitude from the phase and do the phase projection
                tmpa=np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]))
                tmpp=np.angle(np.reshape(recs[:,it+1],[imsizer,imrizec]))#+(np.pi*3/2)
                print(np.max(tmpp),np.min(tmpp))
                tmpaf = tmpa.copy().flatten()
                
                #plt.figure(1111)
                #plt.imshow(tmpp,cmap='gray')  
                #plt.show()
                
                if reglmb == 0: 
                     print("KCT-info: skipping phase proj")
                     tmpptv=tmpp.copy().flatten()
                else:
                     if regtype=='TV':
                          tmpptv=tv_proj(tmpp, mu=0.125,lmb=reglmb,IT=regiter).flatten() #0.1, 15
                     elif regtype=='reg2':
                          tmpptv=reg2_proj(tmpp, alpha=reglmb,niter=regiter).flatten() #0.1, 15
                     else:
                          raise(TypeError)
                
                # combine back the phase and the magnitude
                #tmpptv -= (np.pi*3/2)
                recs[:,it+1] = tmpaf*np.exp(1j*tmpptv)                               #####here
                   
          else:   # the case where you do only data consistency iterations (also iteration 0)
               if not it==0:
                    print('KCT-info: skipping prior proj for the first onlydciters iter.s, doing only phase proj (then maybe DC proj as well) !!!')
                    
               recs[:,it+1]=recs[:,it].copy()
               
               print('current Phase prePsnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec])),255*np.abs(orim),data_range=255))
               print('current Phase Real Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).real),255*np.abs(orim.real),data_range=255))
               print('current Phase Imag Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).imag),255*np.abs(orim.imag),data_range=255))


               # seperate the magnitude from the phase and do the phase projection
               tmpa=np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]))
               tmpp=np.angle(np.reshape(recs[:,it+1],[imsizer,imrizec]))
                
               tmpaf = tmpa.copy().flatten()
                
               if reglmb == 0:
                     print("KCT-info: skipping phase proj")
                     tmpptv=tmpp.copy().flatten()
               else:
                    if regtype=='TV':
                          tmpptv=tv_proj(tmpp, mu=0.125,lmb=reglmb,IT=regiter).flatten() #0.1, 15
                    elif regtype=='reg2':
                          tmpptv=reg2_proj(tmpp, alpha=reglmb,niter=regiter).flatten() #0.1, 15
                    else:
                          raise(TypeError)
                          
               # combine back the phase and the magnitude
               recs[:,it+1] = tmpaf*np.exp(1j*tmpptv)
               #plt.figure(22222)
               #plt.imshow(np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec])),cmap='gray')  
               #plt.show()
          #do the DC projection every 'dcprojiter' iterations    
          if  it < onlydciter+1 or it % dcprojiter == 0: # 
            
              tmp_rec = np.reshape(recs[:,it+1],[imsizer,imrizec])
              #plt.figure(3333333)
              #plt.imshow(np.abs(tmp1),cmap='gray')  
              #plt.show()
              print(np.max(np.abs(tmp_rec)),np.max(255*np.abs(orim)))
              print('current Prior Psnr :',compare_psnr(255*np.abs(tmp_rec),255*np.abs(orim),data_range=255))
              print('current Prior Real Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).real),255*np.abs(orim.real),data_range=255))
              print('current Prior Imag Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).imag),255*np.abs(orim.imag),data_range=255))

              #tmp1 = np.fft.fft2(tmp_rec)
              #print(uspat)
              #tmp1[uspat==1] = us_ksp_r2[uspat==1]
              #tmp1 = np.fft.ifft2(tmp1)
              #recs[:,it+1] = tmp1.flatten()
              #plt.figure(1)
              #plt.imshow(np.abs(tmp1),cmap='gray')  
              #plt.show()
              
           
              tmp1 = np.fft.fft2(np.reshape(recs[:,it+1],[imsizer,imrizec]))*(1-uspat)
              tmp2 = np.fft.fft2(np.reshape(recs[:,it+1],[imsizer,imrizec]))*(uspat)
              
              tmp3= data*uspat#[:,:,np.newaxis]

              #combine the measured data with the projected image affinely (multip=0 for replacing the values)
              tmp=tmp1 + multip*tmp2 + (1-multip)*tmp3
              recs[:,it+1] = np.fft.ifft2(tmp).flatten()   #np.abs(
              print('recs dtype DC ',recs.dtype)
              print('current DC Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec])),255*np.abs(orim),data_range=255))
              print('current DC Real Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).real),255*np.abs(orim.real),data_range=255))
              print('current DC Imag Psnr :',compare_psnr(255*np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]).imag),255*np.abs(orim.imag),data_range=255))
              
              #ftot, f_lik, f_dc = feval(recs[:,it+1])
              
              #print(ftot.shape,f_lik.shape,f_dc.shape)
          x_complex = np.reshape(recs[:,it+1],[256,256])
          psnr = compare_psnr(255*abs(x_complex),255*abs(orim),data_range=255)
          ssim = compare_ssim(abs(x_complex),abs(orim),data_range=1)
          hfen = compare_hfen(abs(x_complex),abs(orim))
          print('current %d image %d iteration PSNR:'%(i,it),psnr,' SSIM :',ssim,' HFEN :',hfen)
          if max_psnr < psnr :
              result_all[i,0] = psnr
              max_psnr = psnr
              result_all[31,0] = sum(result_all[:31,0])/31
              savemat(os.path.join(savepath,'img_{}_Rec_'.format(i)+method+'.mat'),{'data':np.array(np.reshape(recs[:,it+1],[256,256]),dtype=np.complex)})
              cv2.imwrite(os.path.join(savepath,'img_{}_Rec_'.format(i)+method+'.png'), np.array(255*np.abs(np.reshape(recs[:,it+1],[256,256])),dtype=np.uint8))
          if max_ssim < ssim :
              result_all[i,1] = ssim
              max_ssim = ssim
              result_all[31,1] = sum(result_all[:31,1])/31

          if min_hfen > hfen :
              result_all[i,2] = hfen
              min_hfen = hfen
              result_all[31,2] = sum(result_all[:31,2])/31
          write_Data(result_all)
          #print('recs dtype DC ',recs.dtype)
          end = time.time()
          print('{} iteration time is'.format(it),end-start)
     return recs  #YourDatasetModuleHere










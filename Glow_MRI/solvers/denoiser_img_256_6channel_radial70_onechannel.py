import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
import skimage.io as sio
from glow.glow__ import Glow
import json
import matplotlib.pyplot as plt
import os
import warnings
import numbers
from itertools import product
from numpy.lib.stride_tricks import as_strided
import scipy.misc as sm
from PIL import Image
warnings.filterwarnings("ignore")
import scipy.io as io
import time


def Dc_FFt(x,mask,kspace):
    print(x.shape,mask.shape,kspace.shape)
    temp = np.fft.fft2(x)
    temp[mask==1] = kspace[mask==1]
    x = np.fft.ifft2(temp)
    return x
def show(img):
    plt.figure(2)
    plt.imshow(img,cmap='gray')
    plt.show()
def solveDenoising(args):
    if args.prior == 'glow':
        GlowDenoiser(args)
    elif args.prior == 'dcgan':
        GANDenoiser(args)
    else:
        raise "prior not defined correctly"

def show(img):
    plt.figure(2)
    plt.imshow(img,cmap='gray')
    plt.show()
def save_data(args,each_img,psnr,ssim):
    file = open("./radial_70_31_gamma=0.08_one/psnr_mask_radial{}_{}_gamma={}_one.txt".format(args.mask,each_img,args.gamma[0]), "w+")
    for i in range(len(psnr)):
        line = "iter : {}  ".format(i)+"psnr : {}".format(psnr[i])+"  ssim : {}".format(ssim[i])+'\r\n'
        file.write(line)
        if(i == 299):
            line = "iter : {}  ".format(300)+"psnr : {}".format(max(psnr))+"  ssim : {}".format(max(ssim))+'\r\n'
    file.close()
img_shape = 256 

def GlowDenoiser(args):
    loopOver = zip(args.gamma)
    for gamma in loopOver:
        for each_img in range(4,31):
	        skip_to_next  = False # flag to skip to next loop if recovery is fails due to instability
	        n             = img_shape*img_shape*1
	        modeldir      = "./trained_models/%s/glow"%args.model
	        global kspace,mask
	        img = sm.imread('{}.png'.format(each_img+1))
	        #complex_data = np.array(io.loadmat('test1.mat')['img'],dtype=np.complex64)
	        #img = np.array(abs(complex_data),dtype=np.float32)*255.0
	        print(np.max(img),np.min(img))
	        img = np.array(sm.imresize(img,(img_shape,img_shape)),dtype=np.float32)[:,:]/np.max(np.array(img[:,:],dtype=np.float32))
	        sm.imsave('ori_1_test.png',img)
	        mask = io.loadmat('256_mask_under70.mat')['mask']#.format(img_shape,args.mask)   .format(args.mask)
	        print("undersample random mask   :",1-np.sum(mask)/256/256)
	        print(args.mask)
	        ori_img = img
	        #show(mask)
	        x_ori = np.zeros([1,1,img_shape,img_shape],dtype=np.float32)
	        x_ori[:,0:1,:,:] = ori_img#.real         #输入MRI图
	        #x_ori[:,1,:,:] = ori_img#.real          #输入MRI图
	        #x_ori[:,2,:,:] = ori_img#.real         #输入MRI图
	        #x_ori[:,3,:,:] = ori_img#.real         #输入MRI图
	        #x_ori[:,4,:,:] = ori_img#.real          #输入MRI图
	        #x_ori[:,5,:,:] = ori_img#.real         #输入MRI图
	        x_ori = torch.tensor(x_ori,dtype=torch.float,requires_grad=False).to(device=args.device)

	        kspace = np.fft.fft2(img) * mask
	        applymask = np.fft.ifft2(kspace)
	        degrade = np.abs(applymask)
	        sm.imsave('degrade__.png',degrade)
	        # loading glow configurations
	        config_path = modeldir+"/configs.json"
	        with open(config_path, 'r') as f:
		        configs = json.load(f)
	        print(configs)
	        x_test = np.zeros([1,1,img_shape,img_shape],dtype=np.float32) 
	        x_test = torch.tensor(x_test,dtype=torch.float,requires_grad=False)
	        x_test = x_test.clone().to(device=args.device)
	        # regularizor
	        gamma = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
	        n_test = x_test.shape[0]
	        #noise = np.random.normal(0,args.noise_std, size=(n_test,6,img_shape,img_shape))
	        #noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
	        # loading glow model
	        glow = Glow((1,img_shape,img_shape),
			            K=configs["K"],L=configs["L"],
			            coupling=configs["coupling"],
			            n_bits_x=configs["n_bits_x"],
			            nn_init_last_zeros=configs["last_zeros"],
			            device=args.device)
	        glow.load_state_dict(torch.load(modeldir+"/glowmodel.pt"))
	        glow.eval()            

	        # making a forward to record shapes of z's for reverse pass
	        _ = glow(glow.preprocess(torch.zeros_like(x_test)))

	        # initializing z from Gaussian
	        z_sampled = np.random.normal(0,args.init_std,[n_test,n])
	        z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)

	        # selecting optimizer
	        if args.optim == "adam":
		        optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
	        elif args.optim == "lbfgs":
		        optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
	        degrade = torch.tensor(degrade,dtype=torch.float,requires_grad=False)
	        degrade = degrade.clone().to(device=args.device)
	        x_test = torch.tensor(x_test,dtype=torch.float,requires_grad=False)
	        x_test = x_test.clone().to(device=args.device)
	        x_test[:,0:1,:,:] = degrade         #输入MRI图
	        #x_test[:,1,:,:] = degrade         #输入MRI图
	        #x_test[:,2,:,:] = degrade         #输入MRI图
	        #x_test[:,3,:,:] = degrade         #输入MRI图
	        #x_test[:,4,:,:] = degrade         #输入MRI图
	        #x_test[:,5,:,:] = degrade         #输入MRI图
	        print(torch.min(x_test))
	        # to be recorded over iteration
	        psnr_t    = torch.nn.MSELoss().to(device=args.device)
	        #residual  = []
	        # getting test images
	        PSNR_ = []
	        SSIM_ = []
	        PSNR_IMG = []
	        SSIM_IMG = []
	        for k in range(30):
		        print('----',k)
		        for i in range(1):
			        # getting batch of data
			        #print('start this iteration')
			        #degrade = torch.tensor(degrade,dtype=torch.float,requires_grad=False)
			        #degrade = degrade.clone().to(device=args.device)
			        start = time.time()
			
			        # running optimizer steps
			        for t in range(1):
			            def closure():
			                optimizer.zero_grad()
			                z_unflat    = glow.unflatten_z(z_sampled, clone=False)
			                x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
			                x_gen       = glow.postprocess(x_gen,floor_clamp=False)
			                x_noisy     = x_test #+ noise
			                #x_noise_img = x_noisy.detach().cpu().numpy().reshape(-1,img_shape,img_shape)
			                #sm.imsave('noise.png',np.transpose(x_noise_img,[1,2,0]))
			                global residual_t
			                global x_gen_img
			                residual_t  = ((x_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()
			                if args.z_penalty_squared:
			                    z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
			                else:
			                    z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
			                    
			                #x_gen_img = x_gen.detach().cpu().numpy().reshape(-1,img_shape,img_shape)
			                #x_gen_image = np.mean(x_gen_img,0)
			                #print(x_gen_image.shape)
			                #sm.imsave('groundtruth_1.png',x_gen_image)
			                loss_t      = residual_t + z_reg_loss_t
			                psnr        = psnr_t(x_ori, x_gen)
			                psnr        = 10 * np.log10(1 / psnr.item())
			                print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f|gamma=%1.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr,gamma),end="\r")
			                #print('PSNR',compare_psnr(np.array(np.mean(x_gen_img,0),dtype=np.float32),np.array(ori_img,dtype=np.float32),data_range=1))
			                #print('SSIM',compare_ssim(np.array(np.mean(x_gen_img,0),dtype=np.float32),np.array(ori_img,dtype=np.float32),data_range=1))
			                loss_t.backward()
			                return loss_t
			            #try:
			            optimizer.step(closure)
			                #sm.imsave('groundtruth_1.png',np.transpose(x_gen_img,[1,2,0]))
			            #residual.append(residual_t.item())
			            #except:
			            #    skip_to_next = True
			            #    break
			            #if skip_to_next:
			            #    print("\nskipping current loop due to instability or user triggered quit")
			            #    continue
			        if skip_to_next:
			            break
			            
			        # getting recovered and true images
			        #x_test_np  = x_test.data.cpu().numpy().transpose(0,2,3,1)
			        with torch.no_grad():
			            z_unflat   = glow.unflatten_z(z_sampled, clone=False)
			            x_gen      = glow(z_unflat, reverse=True, reverse_clone=False)
			            x_gen      = glow.postprocess(x_gen,floor_clamp=False)
			            x_gen_np   = x_gen.data.cpu().numpy().transpose(0,2,3,1)
			            print(x_gen_np.shape)
			            x_gen_np   = np.clip(x_gen_np,0,1)
			            x_test_np = x_gen_np.squeeze().squeeze()
			            temp = np.fft.fft2(x_test_np)
			            temp[mask==1] = kspace[mask==1]
			            degrade = np.abs(np.fft.ifft2(temp))
			            end = time.time()
			            print('One iteration time is',end-start)
			            psnr_current = compare_psnr(np.array(degrade,dtype=np.float32),np.array(ori_img,dtype=np.float32),data_range=1)
			            ssim_current = compare_ssim(np.array(degrade,dtype=np.float32),np.array(ori_img,dtype=np.float32),data_range=1)
			            
			            PSNR_.append(psnr_current)
			            SSIM_.append(ssim_current)
			            
			            
			            if k == 29:
			                PSNR_IMG.append(psnr_current)
			                SSIM_IMG.append(ssim_current)
			                print('PSNR_mean :',np.mean(PSNR_IMG),'SSIM_mean :',np.mean(SSIM_IMG))
			            save_data(args,each_img,PSNR_,SSIM_)
			            
			            print(degrade.dtype,ori_img.dtype)
			            io.savemat('./radial_70_31_gamma=0.08_one/network_output_mask{}_{}_gamma={}_abs.mat'.format(args.mask,each_img,args.gamma[0]),{'rec':np.array(degrade,dtype=np.float32)})
			            sm.imsave('./radial_70_31_gamma=0.08_one/network_output_mask{}_{}_gamma={}_abs.png'.format(args.mask,each_img,args.gamma[0]),degrade)
			            degrade = torch.tensor(degrade,dtype=torch.float,requires_grad=False)
			            degrade = degrade.clone().to(device=args.device)
			            x_test[:,0:1,:,:] = degrade         #输入MRI图
			            
			            #x_test[:,1,:,:] = degrade          #输入MRI图
			            #x_test[:,2,:,:] = degrade         #输入MRI图
			            #x_test[:,3,:,:] = degrade         #输入MRI图
			            #x_test[:,4,:,:] = degrade          #输入MRI图
			            #x_test[:,5,:,:] = degrade         #输入MRI图
			        #degrade = torch.tensor(degrade,dtype=torch.float,requires_grad=False)
			        #degrade = degrade.clone().to(device=args.device)
			
			        '''del glow
			        glow = Glow((3,img_shape,img_shape),
			            K=configs["K"],L=configs["L"],
			            coupling=configs["coupling"],
			            n_bits_x=configs["n_bits_x"],
			            nn_init_last_zeros=configs["last_zeros"],
			            device=args.device)
			        glow.load_state_dict(torch.load(modeldir+"/glowmodel.pt"))
			        glow.eval()   '''         
	        #save_data(201,max(PSNR_),max(SSIM_))
			        #print('end this iteration')
	        #if skip_to_next:
	        #    print("\nskipping current loop due to instability or user triggered quit")
	        #    continue
			        #degrade = Dc_FFt(x_test_np,mask,kspace)
			        #degrade = degrade.real
			        #Original.append(x_test_np)
			        #Recovered.append(x_gen_np)
			        #Noisy.append(x_noisy_np)
			        #Residual_Curve.append(residual)
			
			        # freeing up memory for second loop
			        #glow.zero_grad()
			        #optimizer.zero_grad()
			        #del x_test, x_gen, optimizer, psnr_t, z_sampled, glow, noise,
			        #torch.cuda.empty_cache()

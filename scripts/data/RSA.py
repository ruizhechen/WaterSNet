# Modified from https://github.com/YanchaoYang/FDA
import torch
import numpy as np

def extract_ampl_phase(fft_im):
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.003):
    _, _, h, w = amp_src.size()
    b=1
    # b = (np.floor(np.amin((h,w))*L)).astype(int)  
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  
    return amp_src

def RSA(src_img, trg_img, L=0.003):
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    return src_in_trg

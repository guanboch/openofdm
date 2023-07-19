# %%
import scipy
import numpy as np
from matplotlib import pyplot as plt
import cmath
import itertools
%matplotlib qt


# %%
polarity = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1,
            1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1,
            1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1,
            -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1,
            -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1,
            -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1,
            1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

PILOT_SUBCARRIES = [-21, -7, 7, 21]
# %%
wave = scipy.fromfile('samples.dat', dtype=scipy.int16)
samples = [complex(i, q) for i, q in zip(wave[::2], wave[1::2])]
lts = np.loadtxt('lts.txt').view(complex)
# %%



fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot([s.real for s in samples[:500]], '-bo')
ax[1].plot([abs(sum([samples[i+j]*samples[i+j+16].conjugate()
    for j in range(0, 48)]))/
    sum([abs(samples[i+j])**2 for j in range(0, 48)])
    for i in range(0, 500)], '-ro')
plt.show()


# %% LSTF field symbol length if 0.8us (16 samples), 10 symbols in total
stf_corr = np.zeros((500,1))
signal = np.asarray(samples)
cor_win_size = 50
for i in range(0, 500):  
    stf_corr[i] = abs(sum(signal[i:i+cor_win_size]*signal[i+16:i+16+cor_win_size].conjugate()))/ sum(abs(signal[i:i+cor_win_size])**2)


plt.figure(2) 
plt.plot(stf_corr)
plt.xlabel('samples')
plt.ylabel('correlation value')
plt.title('correlation of STF for packet detection')
# %% coarse CFO correction using short training field (STF)
alpha_ST = 0 
alpha_ST =np.mean(1/16*np.angle(signal[15:15+160-16].conjugate()*signal[15+16:15+160]))


signal_cfo = np.copy(signal)
lts_offset = 11+160+32
cfo_cor_indx = np.arange(signal_cfo[lts_offset:].shape[0])
signal_cfo[lts_offset:] = signal_cfo[lts_offset:]*np.exp(-1j*cfo_cor_indx*alpha_ST)

# %% fine CFO correction using long training field (LTF)
signal_cfo2 = np.copy(signal_cfo)
beta_ST = np.mean(1/64*np.angle(signal_cfo[11+160+32:11+160+32+64].conjugate()*signal_cfo[11+160+32+64:11+160+32+128]))
sig_offset = lts_offset + 128 #start of guard interval of sig field
cfo_cor_indx2 = np.arange(signal_cfo[sig_offset:].shape[0])
signal_cfo2[sig_offset:] = signal_cfo[sig_offset:]*np.exp(-1j*cfo_cor_indx2*beta_ST)


# %% generate subcarrier mask to indicate unused subcarriers 
sub_mask = np.ones(64)
sub_mask[0:6]=0
sub_mask[64-6:64]=0
sub_mask[32]=0


# %% correlation of LTF field to detect packet start time
lts_corr= np.correlate(signal,lts, mode='valid')
plt.figure(3)
plt.plot(abs(lts_corr[:500]))
lts_loc = np.where(abs(lts_corr[:500])>40000)
lts_loc1= lts_loc[0][0]
init_offset = lts_loc1-160-32 # subtract 32 sample guard interval, 160 (10 symbols) sample short training field

# %%

lts1 = signal[lts_offset:lts_offset+64]
lts2 = signal[lts_offset+64:lts_offset+128]

lts1_f=np.fft.fft(lts1)
lts2_f = np.fft.fft(lts2)
lts_f =np.fft.fft(lts)

lts1_cfo = signal_cfo[lts_offset:lts_offset+64]
lts2_cfo = signal_cfo[lts_offset+64:lts_offset+128]
lts1_cfo_f = np.fft.fft(lts1_cfo)
lts2_cfo_f = np.fft.fft(lts2_cfo)

# long training field only has coarse CFO correction!!!!
lts1_cfo2 = signal_cfo2[lts_offset:lts_offset+64]  #exactly the same as lts1_cfo
lts2_cfo2 = signal_cfo2[lts_offset+64:lts_offset+128]
lts1_cfo2_f = np.fft.fft(lts1_cfo2)
lts2_cfo2_f = np.fft.fft(lts2_cfo2)

plt.figure(31)
plt.plot(np.abs(np.fft.fft(lts1)))
plt.xlabel('subcarrier')
plt.title('spectrum of first LTS')
plt.figure(32)
plt.plot(np.abs(np.fft.fft(lts2)))
plt.xlabel('subcarrier')
plt.title('spectrum of second LTS')

plt.figure(34)
plt.plot(np.abs(lts_f))
plt.xlabel('subcarrier. first half 0 to 10MHz, second half, -10 to 0MHz ')
plt.title('spectrum of template of LTS. real and even. Unused subcarrier is 0')

plt.figure(35)
plt.plot(lts)
plt.xlabel('samples')
plt.title('LTS sequence, real and even')



# %% compute CSI using the two LTF field
# because lts_f field is real and only has -1, 1, and 0 value (0 means unused subcarrier)
# thus we can use multiply instead of divide to calculate CSI
H = 0.5*(lts1_f+lts2_f)*lts_f
H_cfo = 0.5*(lts1_cfo_f+lts2_cfo_f)*lts_f
H_cfo2 = 0.5*(lts1_cfo2_f+lts2_cfo2_f)*lts_f
# H_cfo = H_cfo2!!! long training field only has coarse CFO correction

plt.figure(40)
plt.plot(np.abs(H))
plt.xlabel('subcarrier')
plt.ylabel('amplitude')
plt.title('CSI with close to 0 value at unused subcarrier')
#for some chipset, using division to calculate CSI. we get very large value at unused subcarrier location
H_div = 0.5*(lts1_f+lts2_f)/lts_f
H_cfo_div = 0.5*(lts1_cfo_f+lts2_cfo_f)*lts_f

plt.figure(41)
plt.plot(np.abs(H_div))
plt.xlabel('subcarrier')
plt.ylabel('amplitude')
plt.title('CSI with very large value at unused subcarrier due to division')

# %% symbol equalization with and without CFO correction

plt.close('all')
symbol_start =0
symbol_num = 10
sig_1 = np.zeros((symbol_num,64),dtype=complex) #only equalization
sig_cfo_1 =np.zeros((symbol_num,64),dtype=complex) #equalization + coarse CFO correction
sig_cfo2_1 = np.zeros((symbol_num,64),dtype=complex)#equalization + coarse + fine CFO correction
sig_1f = np.zeros((symbol_num,64),dtype = complex)
sig_cfo_1f = np.zeros((symbol_num,64),dtype =complex)
sig_cfo2_1f = np.zeros((symbol_num,64),dtype =complex) 
all_four_pilot = np.zeros((symbol_num,4),dtype = complex)

polarity_iter = itertools.cycle(polarity)

indx = 0
for i in range(symbol_start, symbol_start +symbol_num):
    p=next(polarity_iter)
    four_pilot = np.array([p, p, p, -p])
    all_four_pilot[indx,:] = four_pilot
    
    
    sig_1[indx,:]= signal[sig_offset+i*80+16:sig_offset+i*80+16+64]
    sig_cfo_1[indx,:]= signal_cfo[sig_offset+i*80+16:sig_offset+i*80+16+64]
    sig_cfo2_1[indx,:]= signal_cfo2[sig_offset+i*80+16:sig_offset+i*80+16+64]
    sig_1f[indx,:] = np.fft.fft(sig_1[indx,:])
    sig_cfo_1f[indx,:] = np.fft.fft(sig_cfo_1[indx,:])
    sig_cfo2_1f[indx,:] = np.fft.fft(sig_cfo2_1[indx,:])
    rfo_1 = np.mean(np.angle(sig_1f[indx,PILOT_SUBCARRIES].conjugate())+np.angle(four_pilot)+np.angle(H[PILOT_SUBCARRIES]))
    rfo_cfo= np.mean(np.angle(sig_cfo_1f[indx,PILOT_SUBCARRIES].conjugate())+np.angle(four_pilot)+np.angle(H_cfo[PILOT_SUBCARRIES]))
    rfo_cfo2 = np.mean(np.angle(sig_cfo2_1f[indx,PILOT_SUBCARRIES].conjugate())+np.angle(four_pilot)+np.angle(H_cfo[PILOT_SUBCARRIES]))
    indx = indx+1


sig_1f_c = sig_1f/H*np.fft.ifftshift(sub_mask)  #only equalization
sig_cfo_1f_c = sig_cfo_1f/H_cfo*np.fft.ifftshift(sub_mask) #equalization + coarse cfo correction
sig_cfo2_1f_c = sig_cfo2_1f/H_cfo*np.fft.ifftshift(sub_mask) #equalization + coarse + fine cfo correction
sig_cfo_rfo_1f_c=sig_cfo_1f/H_cfo*np.exp(1j*rfo_cfo)*np.fft.ifftshift(sub_mask) #equalization + coarse cfo +fine cfo correction + rfo correction
sig_cfo2_rfo_1f_c=sig_cfo2_1f/H_cfo*np.exp(1j*rfo_cfo2)*np.fft.ifftshift(sub_mask) #equalization + coarse cfo +fine cfo correction + rfo correction


sig_1f_c[:,PILOT_SUBCARRIES]= all_four_pilot
sig_cfo_1f_c[:,PILOT_SUBCARRIES]= all_four_pilot  #fill back the exact value of pilot subcarrier
sig_cfo2_1f_c[:,PILOT_SUBCARRIES]= all_four_pilot  #fill back the exact value of pilot subcarrier
sig_cfo_rfo_1f_c[:,PILOT_SUBCARRIES]= all_four_pilot
sig_cfo2_rfo_1f_c[:,PILOT_SUBCARRIES] = all_four_pilot #

# %%
plt.close('all')
sym_idx =np.arange(0,10,1)
plt.figure(71)
plt.plot(sig_cfo_1f_c[sym_idx,:].real.T, sig_cfo_1f_c[sym_idx,:].imag.T,'*')
plt.xlabel('real part')
plt.ylabel('imag part')
plt.title('equalization + coarse cfo correction')

plt.figure(72)
plt.plot(sig_cfo2_1f_c[sym_idx,:].real.T, sig_cfo2_1f_c[sym_idx,:].imag.T,'*')
plt.xlabel('real part')
plt.ylabel('imag part')
plt.title('equalization + coarse cfo +fine cfo correction')

plt.figure(73)
plt.plot(sig_cfo2_rfo_1f_c[sym_idx,:].real.T, sig_cfo2_rfo_1f_c[sym_idx,:].imag.T,'*')
plt.xlabel('real part')
plt.ylabel('imag part')
plt.title('equalization + coarse cfo + fine cfo + rfo correction')

plt.figure(74)
plt.plot(sig_cfo_rfo_1f_c[sym_idx,:].real.T, sig_cfo_rfo_1f_c[sym_idx,:].imag.T,'*')
plt.xlabel('real part')
plt.ylabel('imag part')
plt.title('equalization + coarse cfo + rfo correction')

plt.figure(75)
plt.plot(sig_1f_c[sym_idx,:].real.T, sig_1f_c[sym_idx,:].imag.T,'*')
plt.xlabel('real part')
plt.ylabel('imag part')
plt.title('only equalization')
# %%

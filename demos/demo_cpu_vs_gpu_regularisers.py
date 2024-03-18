#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of CPU implementation against the GPU one

@authors: Daniil Kazantsev, Edoardo Pasca
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from ccpi.filters.regularisers import PatchSelect
from ccpi.supp.qualitymetrics import QualityTools
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'refdata':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################


os.chdir(os.path.join("..", "demos"))
filename = os.path.join("data" ,"lena_gray_512.tif")

# read image
Im = plt.imread(filename)                     
Im = np.asarray(Im, dtype='float32')

Im = Im/255
perc = 0.05
u0 = Im + np.random.normal(loc = 0 ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
u_ref = Im + np.random.normal(loc = 0 ,
                                  scale = 0.01 * Im , 
                                  size = np.shape(Im))

# map the u0 u0->u0>0
# f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________ROF-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of ROF-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02,\
        'number_of_iterations': 1000,\
        'time_marching_parameter': 0.001,\
        'tolerance_constant':0.0}

print ("#############ROF TV CPU####################")
start_time = timeit.default_timer()
infocpu = np.zeros(2, dtype='float32')
rof_cpu = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],
             pars['tolerance_constant'],device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, rof_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
print ("##############ROF TV GPU##################")
start_time = timeit.default_timer()
infogpu = np.zeros(2, dtype='float32')
rof_gpu = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],
             pars['tolerance_constant'],device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, rof_gpu)
pars['rmse'] = Qtools.rmse()

pars['algorithm'] = ROF_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(rof_cpu))
diff_im = abs(rof_cpu - rof_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of FGP-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :400 ,\
        'tolerance_constant':0.0,\
        'methodTV': 0 ,\
        'nonneg': 0}
        
print ("#############FGP TV CPU####################")
start_time = timeit.default_timer()
fgp_cpu =  FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],device='cpu', infovector=infocpu) 

Qtools = QualityTools(Im, fgp_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("##############FGP TV GPU##################")
start_time = timeit.default_timer()
fgp_gpu =  FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],device='gpu', infovector=infogpu) 

Qtools = QualityTools(Im, fgp_gpu)
pars['rmse'] = Qtools.rmse()

pars['algorithm'] = FGP_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(fgp_cpu))
diff_im = abs(fgp_cpu - fgp_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")


#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________PD-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of PD-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : PD_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1500 ,\
        'tolerance_constant':0.0,\
        'methodTV': 0 ,\
        'nonneg': 0,
        'lipschitz_const' : 8}
        
print ("#############PD TV CPU####################")
start_time = timeit.default_timer()
pd_cpu = PD_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['lipschitz_const'],
              pars['methodTV'],
              pars['nonneg'],
              device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, pd_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(pd_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

# set parameters
pars = {'algorithm' : PD_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1500 ,\
        'tolerance_constant':0.0,\
        'methodTV': 0 ,\
        'nonneg': 0,
        'lipschitz_const' : 8}
        
print ("#############PD TV GPU####################")
start_time = timeit.default_timer()
pd_gpu = PD_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['lipschitz_const'],
              pars['methodTV'],
              pars['nonneg'],
              device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, pd_gpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(pd_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(pd_cpu))
diff_im = abs(pd_cpu - pd_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________SB-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of SB-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :250 ,\
        'tolerance_constant':0.0,\
        'methodTV': 0}
        
print ("#############SB-TV CPU####################")
start_time = timeit.default_timer()
sb_cpu = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'], device='cpu', infovector=infocpu)
             

Qtools = QualityTools(Im, sb_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(sb_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("##############SB TV GPU##################")
start_time = timeit.default_timer()
sb_gpu = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'], device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, sb_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = SB_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(sb_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(sb_cpu))
diff_im = abs(sb_cpu - sb_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________LLT-ROF bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of LLT-ROF regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : LLT_ROF, \
        'input' : u0,\
        'regularisation_parameterROF':0.01, \
        'regularisation_parameterLLT':0.0085, \
        'number_of_iterations' : 1000 ,\
        'time_marching_parameter' :0.0001 ,\
        'tolerance_constant':0.0}


print ("#############LLT- ROF CPU####################")
start_time = timeit.default_timer()
lltrof_cpu = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'], device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, lltrof_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(lltrof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("#############LLT- ROF GPU####################")
start_time = timeit.default_timer()
lltrof_gpu = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'], device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, lltrof_gpu)
pars['rmse'] = Qtools.rmse()

pars['algorithm'] = LLT_ROF
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(lltrof_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(lltrof_gpu))
diff_im = abs(lltrof_cpu - lltrof_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________TGV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of TGV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : TGV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'alpha1':1.0,\
        'alpha0':2.0,\
        'number_of_iterations' :1000 ,\
        'LipshitzConstant' :12 ,\
        'tolerance_constant':0.0}
        
print ("#############TGV CPU####################")
start_time = timeit.default_timer()
tgv_cpu = TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'],device='cpu', infovector=infocpu)
             
Qtools = QualityTools(Im, tgv_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(tgv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("##############TGV GPU##################")
start_time = timeit.default_timer()
tgv_gpu = TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'],device='gpu', infovector=infogpu)
                                   
Qtools = QualityTools(Im, tgv_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = TGV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(tgv_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

print ("--------Compare the results--------")
tolerance = 1e-02
diff_im = np.zeros(np.shape(tgv_gpu))
diff_im = abs(tgv_cpu - tgv_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print (f"Arrays do not match! {diff_im.sum()}")
    plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
else:
    print (f"Arrays match {diff_im.sum()}")

N =10
diff = tgv_cpu[:N,:N] - tgv_gpu[:N,:N]
lim = np.max(np.abs(diff))
plt.imshow(diff, vmin=-lim, vmax=lim,  cmap="seismic")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________NDF bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of NDF regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : NDF, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'edge_parameter':0.017,\
        'number_of_iterations' :1500 ,\
        'time_marching_parameter':0.01,\
        'penalty_type':1,\
        'tolerance_constant':0.0}

print ("#############NDF CPU####################")
start_time = timeit.default_timer()
ndf_cpu = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'],device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, ndf_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("##############NDF GPU##################")
start_time = timeit.default_timer()
infogpu = np.zeros(2, dtype='float32')
ndf_gpu = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'],device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, ndf_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = NDF
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(ndf_cpu))
diff_im = abs(ndf_cpu - ndf_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (2D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of Diff4th regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : Diff4th, \
        'input' : u0,\
        'regularisation_parameter':0.8, \
        'edge_parameter':0.02,\
        'number_of_iterations' :1500 ,\
        'time_marching_parameter':0.001,\
        'tolerance_constant':0.0}

print ("#############Diff4th CPU####################")
start_time = timeit.default_timer()
diff4th_cpu = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, diff4th_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(diff4th_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("##############Diff4th GPU##################")
start_time = timeit.default_timer()
diff4th_gpu = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],device='gpu', infovector=infogpu)

Qtools = QualityTools(Im, diff4th_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = Diff4th
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(diff4th_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(diff4th_cpu))
diff_im = abs(diff4th_cpu - diff4th_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-dTV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of FGP-dTV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_dTV, \
        'input' : u0,\
        'refdata' : u_ref,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :500 ,\
        'tolerance_constant':0.0,\
        'eta_const':0.2,\
        'methodTV': 0 ,\
        'nonneg': 0}
        
print ("#############FGP dTV CPU####################")
start_time = timeit.default_timer()
fgp_dtv_cpu = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],device='cpu', infovector=infocpu)

Qtools = QualityTools(Im, fgp_dtv_cpu)
pars['rmse'] = Qtools.rmse()


txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_dtv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("##############FGP dTV GPU##################")
start_time = timeit.default_timer()
fgp_dtv_gpu = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],device='cpu', infovector=infogpu)
Qtools = QualityTools(Im, fgp_dtv_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = FGP_dTV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_dtv_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(fgp_dtv_cpu))
diff_im = abs(fgp_dtv_cpu - fgp_dtv_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____Non-local regularisation bench_________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Comparison of Nonlocal TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

pars = {'algorithm' : PatchSelect, \
        'input' : u0,\
        'searchwindow': 7, \
        'patchwindow': 2,\
        'neighbours' : 15 ,\
        'edge_parameter':0.18}

print ("############## Nonlocal Patches on CPU##################")
start_time = timeit.default_timer()
H_i, H_j, WeightsCPU = PatchSelect(pars['input'], 
              pars['searchwindow'],
              pars['patchwindow'], 
              pars['neighbours'],
              pars['edge_parameter'], device='cpu')
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
#%%
print ("############## Nonlocal Patches on GPU##################")
start_time = timeit.default_timer()
start_time = timeit.default_timer()
H_i, H_j, WeightsGPU = PatchSelect(pars['input'], 
              pars['searchwindow'],
              pars['patchwindow'], 
              pars['neighbours'],
              pars['edge_parameter'],device='gpu')
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(u0))
diff_im = abs(WeightsCPU[0,:,:] - WeightsGPU[0,:,:])
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
#%%
import unittest
import sys
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, DIFF4th
from PIL import Image

class TiffReader(object):
    def imread(self, filename):
        return np.asarray(Image.open(filename))
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
def nrmse(im1, im2):
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(im1.size))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))
    
def rmse(im1, im2):
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(im1.size))
    return rmse
###############################################################################

class TestRegularisers(unittest.TestCase):
    

    def test_ROF_TV_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        
        # set parameters
        pars = {'algorithm': ROF_TV, \
                'input' : u0,\
                'regularisation_parameter':0.04,\
                'number_of_iterations': 1200,\
                'time_marching_parameter': 0.0025        
                }
        print ("#############ROF TV CPU####################")
        start_time = timeit.default_timer()
        rof_cpu = ROF_TV(pars['input'],
                     pars['regularisation_parameter'],
                     pars['number_of_iterations'],
                     pars['time_marching_parameter'],'cpu')
        rms = rmse(Im, rof_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("##############ROF TV GPU##################")
        start_time = timeit.default_timer()
        rof_gpu = ROF_TV(pars['input'], 
                             pars['regularisation_parameter'],
                             pars['number_of_iterations'], 
                             pars['time_marching_parameter'],'gpu')
                             
        rms = rmse(Im, rof_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = ROF_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(rof_cpu))
        diff_im = abs(rof_cpu - rof_gpu)
        diff_im[diff_im > tolerance] = 1

        self.assertLessEqual(diff_im.sum() , 1)
        
    def test_FGP_TV_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        print ("____________FGP-TV bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        
        # set parameters
        pars = {'algorithm' : FGP_TV, \
                'input' : u0,\
                'regularisation_parameter':0.04, \
                'number_of_iterations' :1200 ,\
                'tolerance_constant':0.00001,\
                'methodTV': 0 ,\
                'nonneg': 0 ,\
                'printingOut': 0 
                }
                
        print ("#############FGP TV CPU####################")
        start_time = timeit.default_timer()
        fgp_cpu = FGP_TV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['methodTV'],
                      pars['nonneg'],
                      pars['printingOut'],'cpu')  
                     
                     
        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        
        print ("##############FGP TV GPU##################")
        start_time = timeit.default_timer()
        fgp_gpu = FGP_TV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['methodTV'],
                      pars['nonneg'],
                      pars['printingOut'],'gpu')
                                           
        rms = rmse(Im, fgp_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = FGP_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(fgp_cpu))
        diff_im = abs(fgp_cpu - fgp_gpu)
        diff_im[diff_im > tolerance] = 1

        self.assertLessEqual(diff_im.sum() , 1)

    def test_SB_TV_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("____________SB-TV bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        
        # set parameters
        pars = {'algorithm' : SB_TV, \
                'input' : u0,\
                'regularisation_parameter':0.04, \
                'number_of_iterations' :150 ,\
                'tolerance_constant':1e-05,\
                'methodTV': 0 ,\
                'printingOut': 0 
                }
                
        print ("#############SB-TV CPU####################")
        start_time = timeit.default_timer()
        sb_cpu = SB_TV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['methodTV'],
                      pars['printingOut'],'cpu')  
                     
                     
        rms = rmse(Im, sb_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        
        print ("##############SB TV GPU##################")
        start_time = timeit.default_timer()
        sb_gpu = SB_TV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['methodTV'],
                      pars['printingOut'],'gpu')
                                           
        rms = rmse(Im, sb_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = SB_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(sb_cpu))
        diff_im = abs(sb_cpu - sb_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum(), 1)

    def test_TGV_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        print ("____________TGV bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        
        # set parameters
        pars = {'algorithm' : TGV, \
                'input' : u0,\
                'regularisation_parameter':0.04, \
                'alpha1':1.0,\
                'alpha0':0.7,\
                'number_of_iterations' :250 ,\
                'LipshitzConstant' :12 ,\
                }
                
        print ("#############TGV CPU####################")
        start_time = timeit.default_timer()
        tgv_cpu = TGV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['alpha1'],
                      pars['alpha0'],
                      pars['number_of_iterations'],
                      pars['LipshitzConstant'],'cpu')
                     
        rms = rmse(Im, tgv_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        
        print ("##############TGV GPU##################")
        start_time = timeit.default_timer()
        tgv_gpu = TGV(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['alpha1'],
                      pars['alpha0'],
                      pars['number_of_iterations'],
                      pars['LipshitzConstant'],'gpu')
                                           
        rms = rmse(Im, tgv_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = TGV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(tgv_gpu))
        diff_im = abs(tgv_cpu - tgv_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum() , 1)

    def test_LLT_ROF_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        print ("____________LLT-ROF bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        
        # set parameters
        pars = {'algorithm' : LLT_ROF, \
                'input' : u0,\
                'regularisation_parameterROF':0.04, \
                'regularisation_parameterLLT':0.01, \
                'number_of_iterations' :500 ,\
                'time_marching_parameter' :0.0025 ,\
                }
                
        print ("#############LLT- ROF CPU####################")
        start_time = timeit.default_timer()
        lltrof_cpu = LLT_ROF(pars['input'], 
                      pars['regularisation_parameterROF'],
                      pars['regularisation_parameterLLT'],
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'],'cpu')
        
        rms = rmse(Im, lltrof_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("#############LLT- ROF GPU####################")
        start_time = timeit.default_timer()
        lltrof_gpu = LLT_ROF(pars['input'], 
                      pars['regularisation_parameterROF'],
                      pars['regularisation_parameterLLT'],
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'],'gpu')
        
        rms = rmse(Im, lltrof_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = LLT_ROF
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(lltrof_gpu))
        diff_im = abs(lltrof_cpu - lltrof_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum(), 1)

    def test_Diff4th_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        print ("_______________NDF bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        
        # set parameters
        pars = {'algorithm' : NDF, \
                'input' : u0,\
                'regularisation_parameter':0.06, \
                'edge_parameter':0.04,\
                'number_of_iterations' :1000 ,\
                'time_marching_parameter':0.025,\
                'penalty_type':  1
                }
                
        print ("#############NDF CPU####################")
        start_time = timeit.default_timer()
        ndf_cpu = NDF(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['edge_parameter'], 
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'], 
                      pars['penalty_type'],'cpu')
                     
        rms = rmse(Im, ndf_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        
        print ("##############NDF GPU##################")
        start_time = timeit.default_timer()
        ndf_gpu = NDF(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['edge_parameter'], 
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'], 
                      pars['penalty_type'],'gpu')
                     
        rms = rmse(Im, ndf_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = NDF
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(ndf_cpu))
        diff_im = abs(ndf_cpu - ndf_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum(), 1)

        
    def test_Diff4th_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("___Anisotropic Diffusion 4th Order (2D)____")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        # set parameters
        pars = {'algorithm' : DIFF4th, \
                'input' : u0,\
                'regularisation_parameter':3.5, \
                'edge_parameter':0.02,\
                'number_of_iterations' :500 ,\
                'time_marching_parameter':0.001
                }
        
        print ("#############Diff4th CPU####################")
        start_time = timeit.default_timer()
        diff4th_cpu = DIFF4th(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['edge_parameter'], 
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'],'cpu')
                     
        rms = rmse(Im, diff4th_cpu)
        pars['rmse'] = rms

        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("##############Diff4th GPU##################")
        start_time = timeit.default_timer()
        diff4th_gpu = DIFF4th(pars['input'], 
                      pars['regularisation_parameter'],
                      pars['edge_parameter'], 
                      pars['number_of_iterations'],
                      pars['time_marching_parameter'], 'gpu')
                     
        rms = rmse(Im, diff4th_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = DIFF4th
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(diff4th_cpu))
        diff_im = abs(diff4th_cpu - diff4th_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum() , 1)
        #%%
    def test_FDGdTV_CPU_vs_GPU(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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
        

        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("____________FGP-dTV bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        # set parameters
        pars = {'algorithm' : FGP_dTV, \
                'input' : u0,\
                'refdata' : u_ref,\
                'regularisation_parameter':0.04, \
                'number_of_iterations' :2000 ,\
                'tolerance_constant':1e-06,\
                'eta_const':0.2,\
                'methodTV': 0 ,\
                'nonneg': 0 ,\
                'printingOut': 0 
                }
                
        print ("#############FGP dTV CPU####################")
        start_time = timeit.default_timer()
        fgp_dtv_cpu = FGP_dTV(pars['input'], 
                      pars['refdata'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['eta_const'], 
                      pars['methodTV'],
                      pars['nonneg'],
                      pars['printingOut'],'cpu')
                     
                     
        rms = rmse(Im, fgp_dtv_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("##############FGP dTV GPU##################")
        start_time = timeit.default_timer()
        fgp_dtv_gpu = FGP_dTV(pars['input'], 
                      pars['refdata'], 
                      pars['regularisation_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['eta_const'], 
                      pars['methodTV'],
                      pars['nonneg'],
                      pars['printingOut'],'gpu')
        rms = rmse(Im, fgp_dtv_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = FGP_dTV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(fgp_dtv_cpu))
        diff_im = abs(fgp_dtv_cpu - fgp_dtv_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum(), 1)
        #%%

    def test_cpu_ROF_TV(self):
        #filename = os.path.join(".." , ".." , ".." , "data" ,"testLena.npy")
        
        filename = os.path.join("lena_gray_512.tif")

        plt = TiffReader()
        # read image
        Im = plt.imread(filename)                     
        Im = np.asarray(Im, dtype='float32')
        
        """
        # read noiseless image
        Im = plt.imread(filename)
        Im = np.asarray(Im, dtype='float32')

        Im = Im/255
        """
        tolerance = 1e-05
        rms_rof_exp = 0.006812507 #expected value for ROF model
        rms_fgp_exp = 0.019152347 #expected value for FGP model
        
        # set parameters for ROF-TV
        pars_rof_tv = {'algorithm': ROF_TV, \
                            'input' : Im,\
                            'regularisation_parameter':0.04,\
                            'number_of_iterations': 50,\
                            'time_marching_parameter': 0.0025
                            }
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing ROF-TV (2D, CPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        res = True
        rof_cpu = ROF_TV(pars_rof_tv['input'],
             pars_rof_tv['regularisation_parameter'],
             pars_rof_tv['number_of_iterations'],
             pars_rof_tv['time_marching_parameter'],'cpu')
        rms_rof = rmse(Im, rof_cpu)
        # now compare obtained rms with the expected value
        self.assertLess(abs(rms_rof-rms_rof_exp) , tolerance)
    def test_cpu_FGP_TV(self):
        #filename = os.path.join(".." , ".." , ".." , "data" ,"testLena.npy")
        
        filename = os.path.join("lena_gray_512.tif")

        plt = TiffReader()
        # read image
        Im = plt.imread(filename)                     
        Im = np.asarray(Im, dtype='float32')
        
        """
        # read noiseless image
        Im = plt.imread(filename)
        Im = np.asarray(Im, dtype='float32')

        Im = Im/255
        """
        tolerance = 1e-05
        rms_rof_exp = 0.006812507 #expected value for ROF model
        rms_fgp_exp = 0.019152347 #expected value for FGP model
        
        pars_fgp_tv = {'algorithm' : FGP_TV, \
                            'input' : Im,\
                            'regularisation_parameter':0.04, \
                            'number_of_iterations' :50 ,\
                            'tolerance_constant':1e-08,\
                            'methodTV': 0 ,\
                            'nonneg': 0 ,\
                            'printingOut': 0 
                            }
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing FGP-TV (2D, CPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        fgp_cpu = FGP_TV(pars_fgp_tv['input'], 
              pars_fgp_tv['regularisation_parameter'],
              pars_fgp_tv['number_of_iterations'],
              pars_fgp_tv['tolerance_constant'], 
              pars_fgp_tv['methodTV'],
              pars_fgp_tv['nonneg'],
              pars_fgp_tv['printingOut'],'cpu')  
        rms_fgp = rmse(Im, fgp_cpu)
        # now compare obtained rms with the expected value
        self.assertLess(abs(rms_fgp-rms_fgp_exp) , tolerance)

    def test_gpu_ROF(self):
        #filename = os.path.join(".." , ".." , ".." , "data" ,"testLena.npy")
        filename = os.path.join("lena_gray_512.tif")

        plt = TiffReader()
        # read image
        Im = plt.imread(filename)                     
        Im = np.asarray(Im, dtype='float32')
        
        

        #Im = Im/255
        tolerance = 1e-05
        rms_rof_exp = 0.006812507 #expected value for ROF model
        rms_fgp_exp = 0.019152347 #expected value for FGP model
        
        # set parameters for ROF-TV
        pars_rof_tv = {'algorithm': ROF_TV, \
                            'input' : Im,\
                            'regularisation_parameter':0.04,\
                            'number_of_iterations': 50,\
                            'time_marching_parameter': 0.0025
                            }
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing ROF-TV (2D, GPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        res = True
        rof_gpu = ROF_TV(pars_rof_tv['input'],
             pars_rof_tv['regularisation_parameter'],
             pars_rof_tv['number_of_iterations'],
             pars_rof_tv['time_marching_parameter'],'gpu')
        rms_rof = rmse(Im, rof_gpu)
        # now compare obtained rms with the expected value
        self.assertLess(abs(rms_rof-rms_rof_exp) , tolerance)
    def test_gpu_FGP(self):
        #filename = os.path.join(".." , ".." , ".." , "data" ,"testLena.npy")
        filename = os.path.join("lena_gray_512.tif")

        plt = TiffReader()
        # read image
        Im = plt.imread(filename)                     
        Im = np.asarray(Im, dtype='float32')
        
        

        #Im = Im/255
        tolerance = 1e-05
        rms_rof_exp = 0.006812507 #expected value for ROF model
        rms_fgp_exp = 0.019152347 #expected value for FGP model
        
        # set parameters for FGP-TV
        pars_fgp_tv = {'algorithm' : FGP_TV, \
                            'input' : Im,\
                            'regularisation_parameter':0.04, \
                            'number_of_iterations' :50 ,\
                            'tolerance_constant':1e-08,\
                            'methodTV': 0 ,\
                            'nonneg': 0 ,\
                            'printingOut': 0 
                            }
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing FGP-TV (2D, GPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        fgp_gpu = FGP_TV(pars_fgp_tv['input'], 
              pars_fgp_tv['regularisation_parameter'],
              pars_fgp_tv['number_of_iterations'],
              pars_fgp_tv['tolerance_constant'], 
              pars_fgp_tv['methodTV'],
              pars_fgp_tv['nonneg'],
              pars_fgp_tv['printingOut'],'gpu')  
        rms_fgp = rmse(Im, fgp_gpu)
        # now compare obtained rms with the expected value
        self.assertLess(abs(rms_fgp-rms_fgp_exp) , tolerance)

if __name__ == '__main__':
    unittest.main()

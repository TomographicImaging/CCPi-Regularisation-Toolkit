/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2019 Daniil Kazantsev
 * Copyright 2019 Srikanth Nagella, Edoardo Pasca
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MASK_merge_core.h"
#include "utils.h"

/* A method to ensure connectivity within regions of the segmented image/volume. Here we assume 
 * that the MASK has been obtained using some classification/segmentation method such as k-means or gaussian
 * mixture. Some pixels/voxels have been misclassified and we check the spatial dependences
 * and correct the mask. We check the connectivity using the bresenham line algorithm within the non-local window
 * surrounding the pixel of interest. 
 *
 * Input Parameters:
 * 1. MASK [0:255], the result of some classification algorithm (information-based preferably)
 * 2. The list of classes (e.g. [3,4]) to apply the method. The given order matters. 
 * 3. The total number of classes in the MASK. 
 * 4. The size of the Correction Window inside which the method works. 

 * Output:
 * 1. MASK_upd - the UPDATED MASK where some regions have been corrected (merged) or removed
 * 2. CORRECTEDRegions - The array of the same size as MASK where all regions which were 
 * changed are highlighted and the changes have been counted
 */

float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, int SelClassesList_length, int classesNumb, int CorrectionWindow, int dimX, int dimY, int dimZ)
{
    long i,j,l;
    int counterG, switcher;
    long DimTotal;
    unsigned char *MASK_temp, *ClassesList, CurrClass, temp;
    DimTotal = (long)(dimX*dimY*dimZ);

    /* defines the list for all classes in the mask */
    ClassesList = (unsigned char*) calloc (classesNumb,sizeof(unsigned char));

     /* find which classes (values) are present in the segmented data */
     CurrClass =  MASK[0]; ClassesList[0]= MASK[0]; counterG = 1;
     for(i=0; i<DimTotal; i++) {
       if (MASK[i] != CurrClass) {
          switcher = 1;
          for(j=0; j<counterG; j++) {
            if (ClassesList[j] == MASK[i]) {
              switcher = 0;
              break;
            }}
            if (switcher == 1) {
                CurrClass = MASK[i];
                ClassesList[counterG] = MASK[i];
                /*printf("[%u]\n", ClassesList[counterG]);*/
                counterG++;
              }
        }
        if (counterG == classesNumb) break;
      }
      /* sort from LOW->HIGH the obtained values (classes) */
      for(i=0; i<classesNumb; i++)	{
                  for(j=0; j<classesNumb-1; j++) {
                      if(ClassesList[j] > ClassesList[j+1]) {
                          temp = ClassesList[j+1];
                          ClassesList[j+1] = ClassesList[j];
                          ClassesList[j] = temp;
                      }}}

    MASK_temp = (unsigned char*) calloc (DimTotal,sizeof(unsigned char));

    /* copy given MASK to MASK_upd*/
    copyIm_unchar(MASK, MASK_upd, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
    /********************** PERFORM 2D MASK PROCESSING ************************/
    #pragma omp parallel for shared(MASK,MASK_upd) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    /* STEP1: in a smaller neighbourhood check that the current pixel is NOT an outlier */
    OutiersRemoval2D(MASK, MASK_upd, i, j, (long)(dimX), (long)(dimY));
    }}
    /* copy the updated MASK (clean of outliers) */
    copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));

    for(l=0; l<SelClassesList_length; l++) {
    /*printf("[%u]\n", ClassesList[SelClassesList[l]]);*/
    #pragma omp parallel for shared(MASK_temp,MASK_upd,l) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
      /* The class of the central pixel has not changed, i.e. the central pixel is not an outlier -> continue */
      if (MASK_temp[j*dimX+i] == MASK[j*dimX+i]) {
	    /* !One needs to work with a specific class to avoid overlaps! It is
        crucial to establish relevant classes first (given as an input in SelClassesList) */
       if (MASK_temp[j*dimX+i] == ClassesList[SelClassesList[l]]) {
        /* i = 258; j = 165; */
        Mask_update2D(MASK_temp, MASK_upd, CORRECTEDRegions, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));
        }}
      }}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));
      }
    }
    else {
    /********************** PERFORM 3D MASK PROCESSING ************************/
    
    
    }

    free(MASK_temp);   
    return 0;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float OutiersRemoval2D(unsigned char *MASK, unsigned char *MASK_upd, long i, long j, long dimX, long dimY)
{
  /*if the ROI pixel does not belong to the surrondings, turn it into the surronding*/
  long i_m, j_m, i1, j1, counter;
    counter = 0;
    for(i_m=-1; i_m<=1; i_m++) {
      for(j_m=-1; j_m<=1; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK[j*dimX+i] != MASK[j1*dimX+i1]) counter++;
        }
      }}
      if (counter >= 8) MASK_upd[j*dimX+i] = MASK[j1*dimX+i1];
      return *MASK_upd;
}

float Mask_update2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long i, long j, int CorrectionWindow, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present  */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK_temp[j*dimX+i] != MASK_temp[j1*dimX+i1]) CounterOtherClass++;
        }
      }}
      if (CounterOtherClass > 0) {
      /* the other class is present in the vicinity of CorrectionWindow, continue to STEP 3 */
      /*
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connection.
      Meaning that we're instrested if there are any classes between points A and B that
      does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
              if (MASK_temp[j*dimX+i] == MASK_temp[j1*dimX+i1]) {
                /* A and B points belong to the same class, do STEP 4*/
                /* STEP 4: Run Bresenham line algorithm between A and B points
                and convert all points on the way to the class of A/B.  */
               bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, (long)(dimX), (long)(dimY));
              }
            }
          }}
      }
  return 0;
}
int bresenham2D(int i, int j, int i1, int j1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long dimX, long dimY)
{
                   int n;
                   int x[] = {i, i1};
                   int y[] = {j, j1};
                   int steep = (fabs(y[1]-y[0]) > fabs(x[1]-x[0]));
                   int ystep = 0;

                   //printf("[%i][%i][%i][%i]\n", x[1], y[1], steep, kk) ;
                   //if (steep == 1) {swap(x[0],y[0]); swap(x[1],y[1]);}

                   if (steep == 1) {
                   // swaping
                   int a, b;

                   a = x[0];
                   b = y[0];
                   x[0] = b;
                   y[0] = a;

                   a = x[1];
                   b = y[1];
                   x[1] = b;
                   y[1] = a;
                   }

                   if (x[0] > x[1]) {
                   int a, b;
                   a = x[0];
                   b = x[1];
                   x[0] = b;
                   x[1] = a;

                   a = y[0];
                   b = y[1];
                   y[0] = b;
                   y[1] = a;
                   } //(x[0] > x[1])

                  int delx = x[1]-x[0];
                  int dely = fabs(y[1]-y[0]);
                  int error = 0;
                  int x_n = x[0];
                  int y_n = y[0];
                  if (y[0] < y[1]) {ystep = 1;}
                  else {ystep = -1;}

                  for(n = 0; n<delx+1; n++) {
                       if (steep == 1) {
                        /*printf("[%i][%i][%u]\n", x_n, y_n, MASK[y_n*dimX+x_n]);*/
                        // MASK_upd[x_n*dimX+y_n] = 10;
                        if (MASK[j*dimX+i] != MASK[x_n*dimX+y_n]) {
                        	MASK_upd[x_n*dimX+y_n] = MASK[j*dimX+i];
                        	CORRECTEDRegions[x_n*dimX+y_n] += 1;
                        }
                       }
                       else {
                        // printf("[%i][%i][%u]\n", y_n, x_n, MASK[x_n*dimX+y_n]);
                        // MASK_upd[y_n*dimX+x_n] = 20;
                        if (MASK[j*dimX+i] != MASK[y_n*dimX+x_n]) {
	                        MASK_upd[y_n*dimX+x_n] = MASK[j*dimX+i];
                              	CORRECTEDRegions[y_n*dimX+x_n] += 1;
                        }
                       }
                       x_n = x_n + 1;
                       error = error + dely;

                       if (2*error >= delx) {
                          y_n = y_n + ystep;
                         error = error - delx;
                       } // (2*error >= delx)
                       //printf("[%i][%i][%i]\n", X_new[n], Y_new[n], n) ;
                  } // for(int n = 0; n<delx+1; n++)
                  return 0;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/




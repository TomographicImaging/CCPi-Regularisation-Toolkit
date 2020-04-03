            float t[3];
            float M1 = 0.0f, M2 = 0.0f, M3 = 0.0f;

            l = (j * dimX  + i) * padZ;
        
//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < dimZ; k++) {
                u_upd[l + k] = (u[l + k] + tau * div[l + k] + taulambda * Input[l + k]) / constant;
                udiff[k] = u[l + k] - u_upd[l + k];
                unorm += (udiff[k] * udiff[k]);

#ifdef TNV_LOOP_LAST_I
                gradx_upd[l + k] = 0;
#else
                gradx_upd[l + k] = ((u[l + k + padZ] + tau * div[l + k + padZ] + taulambda * Input[l + k + padZ]) / constant - u_upd[l + k]);
#endif

#ifdef TNV_LOOP_LAST_J
                grady_upd[l + k] = 0;
#else
                grady_upd[l + k] = ((u[l + k + dimX*padZ] + tau * div[l + k + dimX*padZ] + taulambda * Input[l + k + dimX*padZ]) / constant - u_upd[l + k]);
#endif

                gradxdiff[k] = gradx[l + k] - gradx_upd[l + k];
                gradydiff[k] = grady[l + k] - grady_upd[l + k];

                float ubarx = theta1 * gradx_upd[l + k] - theta * gradx[l + k];
                float ubary = theta1 * grady_upd[l + k] - theta * grady[l + k];
//#define TNV_NEW_STYLE                
#ifdef TNV_NEW_STYLE                
                qx_upd[l + k] = qx[l + k] + sigma * ubarx;
                qy_upd[l + k] = qy[l + k] + sigma * ubary;

                float vx = divsigma * qx_upd[l + k]; //+ ubarx
                float vy = divsigma * qy_upd[l + k]; //+ ubary
#else
                float vx = ubarx + divsigma * qx[l + k];
                float vy = ubary + divsigma * qy[l + k];
#endif

                M1 += (vx * vx); M2 += (vx * vy); M3 += (vy * vy);
            }

            coefF(t, M1, M2, M3, sigma, p, q, r);
            
//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < dimZ; k++) {
#ifdef TNV_NEW_STYLE    
                float vx = divsigma * qx_upd[l + k];
                float vy = divsigma * qy_upd[l + k];

                float gx_upd = vx * t[0] + vy * t[1];
                float gy_upd = vx * t[1] + vy * t[2];

                qx_upd[l + k] -= sigma * gx_upd;
                qy_upd[l + k] -= sigma * gy_upd;
#else
                float ubarx = theta1 * gradx_upd[l + k] - theta * gradx[l + k];
                float ubary = theta1 * grady_upd[l + k] - theta * grady[l + k];
                float vx = ubarx + divsigma * qx[l + k];
                float vy = ubary + divsigma * qy[l + k];

                float gx_upd = vx * t[0] + vy * t[1];
                float gy_upd = vx * t[1] + vy * t[2];

                qx_upd[l + k] = qx[l + k] + sigma * (ubarx - gx_upd);
                qy_upd[l + k] = qy[l + k] + sigma * (ubary - gy_upd);
#endif

                float div_upd_val = 0;
#ifndef TNV_LOOP_FIRST_I
                div_upd_val -= qx_upd[l + k - padZ];
#endif

#ifndef TNV_LOOP_FIRST_J
                div_upd_val -= qy_upd[l + k - dimX * padZ];
#endif
#ifndef TNV_LOOP_LAST_I
                div_upd_val += qx_upd[l + k];
#endif
#ifndef TNV_LOOP_LAST_J
                div_upd_val += qy_upd[l + k];
#endif
                div_upd[l + k] = div_upd_val;

                qxdiff = qx[l + k] - qx_upd[l + k];
                qydiff = qy[l + k] - qy_upd[l + k];
                qnorm += (qxdiff * qxdiff + qydiff * qydiff);

                resdual1 += fabs(divsigma * qxdiff - gradxdiff[k]);
                resdual2 += fabs(divsigma * qydiff - gradydiff[k]);
                product += (gradxdiff[k] * qxdiff + gradydiff[k] * qydiff);

#ifndef TNV_LOOP_FIRST_J
                divdiff = div[l + k] - div_upd[l + k];  // Multiple steps... How we compute without history?
                resprimal += fabs(divtau * udiff[k] + divdiff); 
#endif
            }

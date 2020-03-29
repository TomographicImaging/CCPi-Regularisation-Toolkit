    {
            float t[3];
            float M1 = 0.0f, M2 = 0.0f, M3 = 0.0f;
            l = (j * dimX  + i) * padZ;
            m = dimX * padZ;
        
            float *__restrict u_next = u + l + padZ;
            float *__restrict u_current = u + l;
            float *__restrict u_next_row = u + l + m;


            float *__restrict qx_current = qx + l;
            float *__restrict qy_current = qy + l;
            float *__restrict qx_prev = qx + l - padZ;
            float *__restrict qy_prev = qy + l - m;


//  __assume(padZ%16==0);

//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < dimZ; k++) {
                float u_upd = (u[l + k] + tau * div[l + k] + taulambda * Input[l + k]) / constant; // 3 reads
                udiff[k] = u[l + k] - u_upd; // cache 1w
                u[l + k] = u_upd; // 1 write

#ifdef TNV_LOOP_FIRST_J
                udiff0[l + k] = udiff[k];
                div0[l + k] = div[l + k];
#endif

#ifdef TNV_LOOP_LAST_I
                float gradx_upd = 0;
#else
                float u_next_upd = (u[l + k + padZ] + tau * div[l + k + padZ] + taulambda * Input[l + k + padZ]) / constant; // 3 reads
                float gradx_upd = (u_next_upd - u_upd); // 2 reads
#endif

#ifdef TNV_LOOP_LAST_J
                float grady_upd = 0;
#else
                float u_next_row_upd = (u[l + k + m] + tau * div[l + k + m] + taulambda * Input[l + k + m]) / constant; // 3 reads
                float grady_upd = (u_next_row_upd - u_upd); // 1 read
#endif

                gradxdiff[k] = gradx[l + k] - gradx_upd; // 1 read, cache 1w
                gradydiff[k] = grady[l + k] - grady_upd; // 1 read, cache 1w
                gradx[l + k] = gradx_upd; // 1 write
                grady[l + k] = grady_upd; // 1 write
                
                ubarx[k] = gradx_upd - theta * gradxdiff[k]; // cache 1w
                ubary[k] = grady_upd - theta * gradydiff[k]; // cache 1w

                float vx = ubarx[k] + divsigma * qx[l + k]; // 1 read
                float vy = ubary[k] + divsigma * qy[l + k]; // 1 read

                M1 += (vx * vx); M2 += (vx * vy); M3 += (vy * vy);
            }

            coefF(t, M1, M2, M3, sigma, p, q, r);
            
//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < padZ; k++) {
                float vx = ubarx[k] + divsigma * qx_current[k]; // cache 2r
                float vy = ubary[k] + divsigma * qy_current[k]; // cache 2r
                float gx_upd = vx * t[0] + vy * t[1];
                float gy_upd = vx * t[1] + vy * t[2];

                qxdiff = sigma * (ubarx[k] - gx_upd);
                qydiff = sigma * (ubary[k] - gy_upd);
                
                qx_current[k] += qxdiff; // write 1
                qy_current[k] += qydiff; // write 1

                unorm += (udiff[k] * udiff[k]);
                qnorm += (qxdiff * qxdiff + qydiff * qydiff);

                float div_upd = 0;

#ifndef TNV_LOOP_FIRST_I
                div_upd -= qx_prev[k]; // 1 read
#endif
#ifndef TNV_LOOP_FIRST_J
                div_upd -= qy_prev[k]; // 1 read
#endif
#ifndef TNV_LOOP_LAST_I
                div_upd += qx_current[k]; 
#endif
#ifndef TNV_LOOP_LAST_J
                div_upd += qy_current[k]; 
#endif

                divdiff = div[l + k] - div_upd;  // 1 read
                div[l + k] = div_upd; // 1 write

#ifndef TNV_LOOP_FIRST_J
                resprimal += fabs(divtau * udiff[k] + divdiff); 
#endif

                    // We need to have two independent accumulators to allow gcc-autovectorization
                resdual1 += fabs(divsigma * qxdiff + gradxdiff[k]); // cache 1r
                resdual2 += fabs(divsigma * qydiff + gradydiff[k]); // cache 1r
                product -= (gradxdiff[k] * qxdiff + gradydiff[k] * qydiff);
            }
    }
    
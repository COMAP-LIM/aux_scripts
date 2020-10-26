// Compile as
// gcc -shared -std=c11 -O3 -o maplib.so.1 maplib.c
#include <stdio.h>

void makemaps(float* map_full, float* rms_full, int* hit_full,
              float* map_out, float* rms_out, int* hit_out,
              int nfeed, int nband, int nfreq, int nx, int ny){
    for(int i=0; i<nfeed; i++){
        for(int j=0; j<nband; j++){
            for(int k=0; k<nfreq; k++){
                for(int x=0; x<nx; x++){
                    for(int y=0; y<ny; y++){
                        int idx = i*nband*nfreq*nx*ny + j*nfreq*nx*ny + k*nx*ny + x*ny + y;
                        if(hit_full[idx] != 0){
                            map_out[x*ny + y] += map_full[idx]/(rms_full[idx]*rms_full[idx]);
                            rms_out[x*ny + y] += 1.0/(rms_full[idx]*rms_full[idx]);
                            hit_out[x*ny + y] += hit_full[idx];
                        }
                    }
                }
            }
        }
    }
}
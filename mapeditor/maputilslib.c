// Compile as
// gcc -shared maputilslib.c -o maputilslib.so.1
#include <stdio.h>
#include <time.h>
#include <math.h>

// Coadd functions
void coadd4D(float* map1, int* nhit1, float* rms1, 
             float* map2, int* nhit2, float* rms2,
             float* map,  int* nhit,  float* rms,
             int n0,      int n1,     int n2, 
             int n3                
             ){
    /*
    Function looping through 4D datasets of two input files and coadding them, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with coadded data.
    nhit: int*
        Output nhit dataset to be filled with coadded data.
    rms: float*
        Output rms dataset to be filled with coadded data.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n2 * n3;     // Precomputing index factors for flattened index
    int prod1 = prod * n1;
    int idx;                    // Index of flattened arrays
    float inv_var1, inv_var2;   // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 4D datasets and coadding them.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    idx = prod1 * i + prod * j + n3 * k + l;
                    nhit_times_nhit = nhit1[idx];
                    nhit_times_nhit *= nhit2[idx];
                    if (nhit_times_nhit > 0){
                        inv_var1  = 1.0 / (rms1[idx] * rms1[idx]);
                        inv_var2  = 1.0 / (rms2[idx] * rms2[idx]);
                        map[idx]  = map1[idx] * inv_var1 + map2[idx] * inv_var2;
                        map[idx] /=  inv_var1 + inv_var2;

                        nhit[idx] = nhit1[idx] + nhit2[idx];
                        rms[idx] = 1 / sqrt(inv_var1 + inv_var2);
                    }
                    else {
                        map[idx] = 0;
                        nhit[idx] = 0;
                        rms[idx] = 0;
                    }       
                }
            }
        }
    }
}

void coadd5D(float* map1, int* nhit1, float* rms1, 
             float* map2, int* nhit2, float* rms2,
             float* map,  int* nhit,  float* rms,
             int n0,      int n1,     int n2, 
             int n3,      int n4                
             ){
    /*
    Function looping through 5D datasets of two input files and coadding them, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with coadded data.
    nhit: int*
        Output nhit dataset to be filled with coadded data.
    rms: float*
        Output rms dataset to be filled with coadded data.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n3 * n4;     // Precomputing index factors for flattened index
    int prod1 = prod * n2;
    int prod2 =  prod1 * n1;
    int idx;                    // Index of flattened arrays
    float inv_var1, inv_var2;   // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 5D datasets and coadding them.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        idx = prod2 * i + prod1 * j + prod * k + n4 * l + m;
                        nhit_times_nhit = nhit1[idx];
                        nhit_times_nhit *= nhit2[idx];
                        if (nhit_times_nhit > 0){
                            inv_var1  = 1.0 / (rms1[idx] * rms1[idx]);
                            inv_var2  = 1.0 / (rms2[idx] * rms2[idx]);
                            map[idx]  = map1[idx] * inv_var1 + map2[idx] * inv_var2;
                            map[idx] /=  inv_var1 + inv_var2;

                            nhit[idx] = nhit1[idx] + nhit2[idx];
                            rms[idx] = 1 / sqrt(inv_var1 + inv_var2);
                        }
                        else {
                            map[idx] = 0;
                            nhit[idx] = 0;
                            rms[idx] = 0;
                        }       
                    }
                }
            }
        }
    }
}

void coadd6D(float* map1, int* nhit1, float* rms1, 
             float* map2, int* nhit2, float* rms2,
             float* map,  int* nhit,  float* rms,
             int n0,      int n1,     int n2, 
             int n3,      int n4,     int n5){
    /*
    Function looping through 6D datasets of two input files and coadding them, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with coadded data.
    nhit: int*
        Output nhit dataset to be filled with coadded data.
    rms: float*
        Output rms dataset to be filled with coadded data.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n4 * n5;     // Precomputing index factors for flattened index
    int prod1 = prod * n3;
    int prod2 =  prod1 * n2;
    int prod3 =  prod2 * n1;
    int idx;                    // Index of flattened arrays
    float inv_var1, inv_var2;   // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 6D datasets and coadding them.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        for(int n=0; n < n5; n++){
                            idx = prod3 * i + prod2 * j + prod1 * k + prod * l + n5 * m + n;
                            nhit_times_nhit = nhit1[idx];
                            nhit_times_nhit *= nhit2[idx];
                            if (nhit_times_nhit > 0){
                                inv_var1  = 1.0 / (rms1[idx] * rms1[idx]);
                                inv_var2  = 1.0 / (rms2[idx] * rms2[idx]);
                                map[idx]  = map1[idx] * inv_var1 + map2[idx] * inv_var2;
                                map[idx] /=  inv_var1 + inv_var2;

                                nhit[idx] = nhit1[idx] + nhit2[idx];
                                rms[idx] = 1 / sqrt(inv_var1 + inv_var2);
                            }
                            else {
                                map[idx] = 0;
                                nhit[idx] = 0;
                                rms[idx] = 0;
                            }
                        }      
                    }
                }
            }
        }
    }
}

// Subtract functions
void subtract4D(float* map1, int* nhit1, float* rms1, 
                float* map2, int* nhit2, float* rms2,
                float* map,  int* nhit,  float* rms,
                int n0,      int n1,     int n2, 
                int n3                
                ){
    /*
    Function looping through 4D datasets of two input files and subtracting them
    from each other, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n2 * n3;     // Precomputing index factors for flattened index
    int prod1 = prod * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    
    long int nhit_times_nhit; 
    
    // Looping through 4D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    idx = prod1 * i + prod * j + n3 * k + l;
                    nhit_times_nhit = nhit1[idx];
                    nhit_times_nhit *= nhit2[idx];
                    if (nhit_times_nhit > 0){
                        var1  = rms1[idx] * rms1[idx];
                        var2  = rms2[idx] * rms2[idx];
                        map[idx]  = map1[idx] - map2[idx];
                        nhit[idx] = nhit1[idx] + nhit2[idx];
                        rms[idx]  =  sqrt(var1 + var2);
                    }
                    else {
                        map[idx] = 0;
                        nhit[idx] = 0;
                        rms[idx] = 0;
                    }       
                }
            }
        }
    }
}

void subtract5D(float* map1, int* nhit1, float* rms1, 
             float* map2, int* nhit2, float* rms2,
             float* map,  int* nhit,  float* rms,
             int n0,      int n1,     int n2, 
             int n3,      int n4                
             ){
    /*
    Function looping through 5D datasets of two input files and subtracting them
    from each other, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n3 * n4;     // Precomputing index factors for flattened index
    int prod1 = prod * n2;  
    int prod2 =  prod1 * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 5D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        idx = prod2 * i + prod1 * j + prod * k + n4 * l + m;
                        nhit_times_nhit = nhit1[idx];
                        nhit_times_nhit *= nhit2[idx];
                        if (nhit_times_nhit > 0){
                            var1        = rms1[idx] * rms1[idx];
                            var2        = rms2[idx] * rms2[idx];
                            map[idx]    = map1[idx] - map2[idx];
                            
                            nhit[idx]   = nhit1[idx] + nhit2[idx];
                            rms[idx]    =  sqrt(var1 + var2);
                        }
                        else {
                            map[idx]    = 0;
                            nhit[idx]   = 0;
                            rms[idx]    = 0;
                        }       
                    }
                }
            }
        }
    }
}

void subtract6D(float* map1, int* nhit1, float* rms1, 
             float* map2, int* nhit2, float* rms2,
             float* map,  int* nhit,  float* rms,
             int n0,      int n1,     int n2, 
             int n3,      int n4,     int n5){
    /*
    Function looping through 5D datasets of two input files and subtracting them
    from each other, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n4 * n5;     // Precomputing index factors for flattened index
    int prod1 = prod * n3;
    int prod2 =  prod1 * n2;
    int prod3 =  prod2 * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 5D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        for(int n=0; n < n5; n++){
                            idx = prod3 * i + prod2 * j + prod1 * k + prod * l + n5 * m + n;
                            nhit_times_nhit = nhit1[idx];
                            nhit_times_nhit *= nhit2[idx];
                            if (nhit_times_nhit > 0){
                                var1  = rms1[idx] * rms1[idx];
                                var2  = rms2[idx] * rms2[idx];
                                map[idx]  = map1[idx] - map2[idx];

                                nhit[idx] = nhit1[idx] + nhit2[idx];
                                rms[idx]  =  sqrt(var1 + var2);
                            }
                            else {
                                map[idx] = 0;
                                nhit[idx] = 0;
                                rms[idx] = 0;
                            }
                        }      
                    }
                }
            }
        }
    }
}

// Subtract functions
void add4D(float* map1, int* nhit1, float* rms1, 
           float* map2, int* nhit2, float* rms2,
           float* map,  int* nhit,  float* rms,
           int n0,      int n1,     int n2, 
           int n3                
           ){
    /*
    Function looping through 4D datasets of two input files and adding them, 
    to return the product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n2 * n3;     // Precomputing index factors for flattened index
    int prod1 = prod * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 4D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    idx = prod1 * i + prod * j + n3 * k + l;
                    nhit_times_nhit = nhit1[idx];
                    nhit_times_nhit *= nhit2[idx];
                    if (nhit_times_nhit > 0){
                        var1  = rms1[idx] * rms1[idx];
                        var2  = rms2[idx] * rms2[idx];
                        map[idx]  = map1[idx] + map2[idx];
                        nhit[idx] = nhit1[idx] + nhit2[idx];
                        rms[idx]  =  sqrt(var1 + var2);
                    }
                    else {
                        map[idx] = 0;
                        nhit[idx] = 0;
                        rms[idx] = 0;
                    }       
                }
            }
        }
    }
}

void add5D(float* map1, int* nhit1, float* rms1, 
           float* map2, int* nhit2, float* rms2,
           float* map,  int* nhit,  float* rms,
           int n0,      int n1,     int n2, 
           int n3,      int n4                
           ){
    /*
    Function looping through 5D datasets of two input files and adding them, 
    to return the product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n3 * n4;     // Precomputing index factors for flattened index
    int prod1 = prod * n2;  
    int prod2 =  prod1 * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 5D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        idx = prod2 * i + prod1 * j + prod * k + n4 * l + m;
                        nhit_times_nhit = nhit1[idx];
                        nhit_times_nhit *= nhit2[idx];
                        if (nhit_times_nhit > 0){
                            var1        = rms1[idx] * rms1[idx];
                            var2        = rms2[idx] * rms2[idx];
                            map[idx]    = map1[idx] + map2[idx];
                            
                            nhit[idx]   = nhit1[idx] + nhit2[idx];
                            rms[idx]    =  sqrt(var1 + var2);
                        }
                        else {
                            map[idx]    = 0;
                            nhit[idx]   = 0;
                            rms[idx]    = 0;
                        }       
                    }
                }
            }
        }
    }
}

void add6D(float* map1, int* nhit1, float* rms1, 
           float* map2, int* nhit2, float* rms2,
           float* map,  int* nhit,  float* rms,
           int n0,      int n1,     int n2, 
           int n3,      int n4,     int n5){
    /*
    Function looping through 5D datasets of two input files and adding them, 
    to return the product by call-by-pointer.

    Parameters
    ------------------------------
    map1: float*
        First input map dataset.
    nhit1: int*
        First input nhit dataset.
    rms1: float*
        First input rms dataset.
    map2: float*
        Second input map2 dataset.
    nhit2: int*
        Second input nhit2 dataset.
    rms2: float*
        Second input rms2 dataset.
    map: float*
        Output map dataset to be filled with subtraction.
    nhit: int*
        Output nhit dataset to be filled with subtraction.
    rms: float*
        Output rms dataset to be filled with subtraction.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    
    ------------------------------
    */
    int prod = n4 * n5;     // Precomputing index factors for flattened index
    int prod1 = prod * n3;
    int prod2 =  prod1 * n2;
    int prod3 =  prod2 * n1;
    int idx;                // Index of flattened arrays
    float var1, var2;       // Inverse variances
    long int nhit_times_nhit; 

    // Looping through 5D datasets and performing subtractions.
    for(int i=0; i < n0; i++){
        for(int j=0; j < n1; j++){
            for(int k=0; k < n2; k++){
                for(int l=0; l < n3; l++){
                    for(int m=0; m < n4; m++){
                        for(int n=0; n < n5; n++){
                            idx = prod3 * i + prod2 * j + prod1 * k + prod * l + n5 * m + n;
                            nhit_times_nhit = nhit1[idx];
                            nhit_times_nhit *= nhit2[idx];
                            if (nhit_times_nhit > 0){
                                var1  = rms1[idx] * rms1[idx];
                                var2  = rms2[idx] * rms2[idx];
                                map[idx]  = map1[idx] + map2[idx];

                                nhit[idx] = nhit1[idx] + nhit2[idx];
                                rms[idx]  =  sqrt(var1 + var2);
                            }
                            else {
                                map[idx] = 0;
                                nhit[idx] = 0;
                                rms[idx] = 0;
                            }
                        }      
                    }
                }
            }
        }
    }
}


// Upgrading functions on pixel level
void dgradeXY4D(float* map_h, int* nhit_h, float* rms_h, 
              float* map_l, int* nhit_l, float* rms_l,
              int n0,       int n1,      int n2, 
              int n3,       int N2,      int N3,       
              int num){
    /*
    Function looping through 4D datasets of one input file and transforms the dataset from
    a high-resolution pixel grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new pixel image.
    num: int
        Number of pixels to merge along each dimension.
    
    ------------------------------
    */             
    // Precomputing index factors for flattened indies
    int p_h = n2 * n3;      
    int p_h1 = p_h * n1;
    int idx_h;
    
    int p_l = N2 * N3;
    int p_l1 = p_l * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, a, b;

    // Looping through datasets to co-merge neighboring pixels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(a = 0; a < N2; a++){
                for(b = 0; b < N3; b++){
                    idx_l = p_l1 * i + p_l * j + N3 * a + b;
                    inv_var1_sum = 0;
                    
                    for(k = a * num; k < (a + 1) * num; k++){
                        for(l = b * num; l < (b + 1) * num; l++){
                            idx_h = p_h1 * i + p_h * j + n3 * k + l;
                            if (nhit_h[idx_h] > 0){
                                inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                nhit_l[idx_l]    += nhit_h[idx_h];
                            }
                            else{
                                inv_var1    = 0;
                            }
                            inv_var1_sum   += inv_var1;
                        }
                    }
                    if (nhit_l[idx_l] > 0){
                        map_l[idx_l] /= inv_var1_sum;
                        rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                    }
                    else{
                        map_l[idx_l] = 0;
                        rms_l[idx_l] = 0;
                    }
                }      
            }
        }
    }
}

void dgradeXY5D(float* map_h, int* nhit_h, float* rms_h, 
              float* map_l, int* nhit_l, float* rms_l,
              int n0, int n1, int n2,
              int n3, int n4, int N3,
              int N4, int num){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a high-resolution pixel grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new pixel image.
    num: int
        Number of pixels to merge along each dimension.
    
    ------------------------------
    */  

    // Precomputing index factors for flattened indies
    int p_h = n3 * n4;
    int p_h1 = p_h * n2;
    int p_h2 =  p_h1 * n1;
    int idx_h;
    
    int p_l = N3 * N4;
    int p_l1 = p_l * n2;
    int p_l2 =  p_l1 * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, a, b;

    // Looping through datasets to co-merge neighboring pixels.
    for(i=0; i < n0; i++){
        for(j = 0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(a = 0; a < N3; a++){
                    for(b = 0; b < N4; b++){
                            
                        idx_l = p_l2 * i + p_l1 * j + p_l * k + N4 * a + b;
                        inv_var1_sum = 0;
                        
                        for(l = a * num; l < (a + 1) * num; l++){
                            for(m = b * num; m < (b + 1) * num; m++){
                                idx_h = p_h2 * i + p_h1 * j + p_h * k + n4 * l + m;
                                
                                if (nhit_h[idx_h] > 0){
                                    inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                    map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                    nhit_l[idx_l]    += nhit_h[idx_h];
                                }
                                else{
                                    inv_var1    = 0;
                                }
                                inv_var1_sum   += inv_var1;
                            }
                        }
                        if (nhit_l[idx_l] > 0){
                            map_l[idx_l] /= inv_var1_sum;
                            rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                        }
                        else{
                            map_l[idx_l] = 0;
                            rms_l[idx_l] = 0;
                        }
                    }      
                }
            }
        }
    }
}

void dgradeXY6D(float* map_h, int* nhit_h, float* rms_h, 
              float* map_l,  int* nhit_l,  float* rms_l,
              int n0,      int n1,     int n2, 
              int n3,      int n4,     int n5,
              int N4,      int N5,     int num){
    /*
    Function looping through 6D datasets of one input file and transforms the dataset from
    a high-resolution pixel grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new pixel image.
    num: int
        Number of pixels to merge along each dimension.
    
    ------------------------------
    */  

    // Precomputing index factors for flattened indies
    int p_h = n4 * n5;
    int p_h1 = p_h * n3;
    int p_h2 =  p_h1 * n2;
    int p_h3 =  p_h2 * n1;
    int idx_h;

    int p_l = N4 * N5;
    int p_l1 = p_l * n3;
    int p_l2 = p_l1 * n2;
    int p_l3 = p_l2 * n1;
    int idx_l;
   
    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, n, a, b;
    
    // Looping through datasets to co-merge neighboring pixels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(l=0; l < n3; l++){
                    for(a = 0; a < N4; a++){
                        for(b = 0; b < N5; b++){
                            
                            idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * l + N5 * a + b;
                            inv_var1_sum = 0;

                            for(m = a * num; m < (a + 1) * num; m++){
                                for(n = b * num; n < (b + 1) * num; n++){
                                    idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + n5 * m + n;
                                    if (nhit_h[idx_h] > 0){
                                        inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                        map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                        nhit_l[idx_l]    += nhit_h[idx_h];
                                    }
                                    else{
                                        inv_var1    = 0;
                                    }
                                    inv_var1_sum   += inv_var1;
                                }
                            }
                            if (nhit_l[idx_l] > 0){
                                map_l[idx_l] /= inv_var1_sum;
                                rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                            }
                            else{
                                map_l[idx_l] = 0;
                                rms_l[idx_l] = 0;
                            }
                        }      
                    }
                }
            }
        }
    }
}

// Upgrading functions on frequency channel level
void dgradeZ4D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l, float* rms_l,
               int n0,       int n1,      int n2, 
               int n3,       int N1,      int num){
    /*
    Function looping through 4D datasets of one input file and transforms the dataset from
    a high-resolution frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new length of frequency axis.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  

    // Precomputing index factors for flattened indies                   
    int p_h = n2 * n3;
    int p_h1 = p_h * n1;
    int idx_h;
    
    int p_l = n2 * n3;
    int p_l1 = p_l * N1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, a;

    // Looping through datasets to co-merge neighboring channels.
    for(i=0; i < n0; i++){
        for(a = 0; a < N1; a++){

            for(k = 0; k < n2; k++){
                for(l = 0; l < n3; l++){
                    inv_var1_sum = 0;
                    for(j = a * num; j < (a + 1) * num; j++){    
                        idx_l = p_l1 * i + p_l * a + n3 * k + l;
                        idx_h = p_h1 * i + p_h * j + n3 * k + l;
                        if (nhit_h[idx_h] > 0){
                            inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                            map_l[idx_l]     += map_h[idx_h] * inv_var1;
                            nhit_l[idx_l]    += nhit_h[idx_h];
                        }
                        else{
                            inv_var1    = 0;
                        }
                        inv_var1_sum   += inv_var1;
                    }
                    if (nhit_l[idx_l] > 0){
                        map_l[idx_l] /= inv_var1_sum;
                        rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                    }
                    else{
                        map_l[idx_l] = 0;
                        rms_l[idx_l] = 0;
                    }
                }
            }
        }
    }
}

void dgradeZ5D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l, float* rms_l,
               int n0, int n1, int n2,
               int n3, int n4, int N2,
               int num){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a high-resolution frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new length of frequency axis.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                   
    int p_h = n3 * n4;
    int p_h1 = p_h * n2;
    int p_h2 =  p_h1 * n1;
    int idx_h;
    
    int p_l = n3 * n4;
    int p_l1 = p_l * N2;
    int p_l2 =  p_l1 * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, a;

    // Looping through datasets to co-merge neighboring channels.
    for(i=0; i < n0; i++){
        for(j = 0; j < n1; j++){
            for(a = 0; a < N2; a++){

                for(l = 0; l < n3; l++){
                    for(m = 0; m < n4; m++){
                        inv_var1_sum = 0;
                        for(k = a * num; k < (a + 1) * num; k++){    
                            idx_l = p_l2 * i + p_l1 * j + p_l * a + n4 * l + m;
                            idx_h = p_h2 * i + p_h1 * j + p_h * k + n4 * l + m;
                            
                            if (nhit_h[idx_h] > 0){
                                inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                nhit_l[idx_l]    += nhit_h[idx_h];
                            }
                            else{
                                inv_var1    = 0;
                            }
                            inv_var1_sum   += inv_var1;
                        }
                        if (nhit_l[idx_l] > 0){
                            map_l[idx_l] /= inv_var1_sum;
                            rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                        }
                        else{
                            map_l[idx_l] = 0;
                            rms_l[idx_l] = 0;
                        }
                    }     
                }
            }
        }
    }
}

void dgradeZ6D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l,  float* rms_l,
               int n0,       int n1,     int n2, 
               int n3,       int n4,     int n5,
               int N3,       int num){
    /*
    Function looping through 6D datasets of one input file and transforms the dataset from
    a high-resolution frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis lenght of each dimension of the new length of frequency axis.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies
    int p_h = n4 * n5;
    int p_h1 = p_h * n3;
    int p_h2 =  p_h1 * n2;
    int p_h3 =  p_h2 * n1;
    int idx_h;

    int p_l = n4 * n5;
    int p_l1 = p_l * N3;
    int p_l2 = p_l1 * n2;
    int p_l3 = p_l2 * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, n, a;

    // Looping through datasets to co-merge neighboring channels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(a = 0; a < N3; a++){
                    
                    for(m = 0; m < n4; m++){
                        for(n = 0; n < n5; n++){
                            inv_var1_sum = 0;
                            for(l = a * num; l < (a + 1) * num; l++){
                                idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * a + n5 * m + n;
                                idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + n5 * m + n;
                                if (nhit_h[idx_h] > 0){
                                    inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                    map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                    nhit_l[idx_l]    += nhit_h[idx_h];
                                }
                                else{
                                    inv_var1    = 0;
                                }
                                inv_var1_sum   += inv_var1;
                            }
                            if (nhit_l[idx_l] > 0){
                                map_l[idx_l] /= inv_var1_sum;
                                rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                            }
                            else{
                                map_l[idx_l] = 0;
                                rms_l[idx_l] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Upgrading functions on pixel and frequency channel level
void dgradeXYZ4D(float* map_h, int* nhit_h, float* rms_h, 
                 float* map_l, int* nhit_l, float* rms_l,
                 int n0,       int n1,      int n2, 
                 int n3,       int N1,      int N2,       
                 int N3,       int numZ,    int numXY){
    /*
    Function looping through 4D datasets of one input file and transforms the dataset from
    a high-resolution pixel-frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel and frequency axes.
    numZ: int
        Number of channels to split into along each dimension.
    numXY: int
        Number of pixels to split into along each dimension.
    ------------------------------
    */  
    // Precomputing index factors for flattened indies
    int p_h = n2 * n3;
    int p_h1 = p_h * n1;
    int idx_h;
    
    int p_l = N2 * N3;
    int p_l1 = p_l * N1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, x, y, z;

    // Looping through datasets to co-merge neighboring pixels and channels.
    for(i=0; i < n0; i++){
        for(z = 0; z < N1; z++){
            for(x = 0; x < N2; x++){
                for(y = 0; y < N3; y++){

                    inv_var1_sum = 0;
                    for(j = z * numZ; j < (z + 1) * numZ; j++){
                        for(k = x * numXY; k < (x + 1) * numXY; k++){
                            for(l = y * numXY; l < (y + 1) * numXY; l++){
                                idx_l = p_l1 * i + p_l * z + N3 * x + y;
                                idx_h = p_h1 * i + p_h * j + n3 * k + l;
                    
                                if (nhit_h[idx_h] > 0){
                                    inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                    map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                    nhit_l[idx_l]    += nhit_h[idx_h];
                                }
                                else{
                                    inv_var1    = 0;
                                }
                                inv_var1_sum   += inv_var1;
                            }
                        }
                    }
                    if (nhit_l[idx_l] > 0){
                        map_l[idx_l] /= inv_var1_sum;
                        rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                    }
                    else{
                        map_l[idx_l] = 0;
                        rms_l[idx_l] = 0;
                    }
                }        
            }
        }
    }   
}

void dgradeXYZ5D(float* map_h, int* nhit_h, float* rms_h, 
                 float* map_l, int* nhit_l, float* rms_l,
                 int n0,       int n1,      int n2, 
                 int n3,       int n4,      int N2,       
                 int N3,       int N4,      int numZ,     
                 int numXY){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a high-resolution pixel-frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel and frequency axes.
    numZ: int
        Number of channels to split into along each dimension.
    numXY: int
        Number of pixels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                 
    int p_h = n3 * n4;
    int p_h1 = p_h * n2;
    int p_h2 =  p_h1 * n1;
    int idx_h;
    
    int p_l = N3 * N4;
    int p_l1 = p_l * N2;
    int p_l2 =  p_l1 * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, x, y, z;

    // Looping through datasets to co-merge neighboring pixels and channels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(z = 0; z < N2; z++){
                for(x = 0; x < N3; x++){
                    for(y = 0; y < N4; y++){

                        inv_var1_sum = 0;
                        for(k = z * numZ; k < (z + 1) * numZ; k++){
                            for(l = x * numXY; l < (x + 1) * numXY; l++){
                                for(m = y * numXY; m < (y + 1) * numXY; m++){
                                    idx_l = p_l2 * i + p_l1 * j + p_l * z + N4 * x + y;
                                    idx_h = p_h2 * i + p_h1 * j + p_h * k + n4 * l + m;
                        
                                    if (nhit_h[idx_h] > 0){
                                        inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                        map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                        nhit_l[idx_l]    += nhit_h[idx_h];
                                    }
                                    else{
                                        inv_var1    = 0;
                                    }
                                    inv_var1_sum   += inv_var1;
                                }
                            }
                        }
                        if (nhit_l[idx_l] > 0){
                            map_l[idx_l] /= inv_var1_sum;
                            rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                        }
                        else{
                            map_l[idx_l] = 0;
                            rms_l[idx_l] = 0;
                        }
                    }        
                }
            }
        }
    }   
}

void dgradeXYZ6D(float* map_h, int* nhit_h, float* rms_h, 
                 float* map_l, int* nhit_l, float* rms_l,
                 int n0,       int n1,      int n2, 
                 int n3,       int n4,      int n5,
                 int N3,       int N4,      int N5, 
                 int numZ,     int numXY){
    /*
    Function looping through 6D datasets of one input file and transforms the dataset from
    a high-resolution pixel-frequency grid to a low-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Input map high-resolution dataset.
    nhit_h: int*
        Input nhit high-resolution dataset.
    rms_h: float*
        Input rms high-resolution dataset.
    map_l: float*
        Output map low-resolution dataset.
    nhit_l: int*
        Output nhit low-resolution dataset.
    rms_l: float*
        Output rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel and frequency axes.
    numZ: int
        Number of channels to split into along each dimension.
    numXY: int
        Number of pixels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                 
    int p_h = n4 * n5;
    int p_h1 = p_h * n3;
    int p_h2 =  p_h1 * n2;
    int p_h3 =  p_h2 * n1;
    int idx_h;

    int p_l = N4 * N5;
    int p_l1 = p_l * N3;
    int p_l2 = p_l1 * n2;
    int p_l3 = p_l2 * n1;
    int idx_l;

    float inv_var1, inv_var1_sum;
    int i, j, k ,l, m, n, x, y, z;

    // Looping through datasets to co-merge neighboring pixels and channels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(z = 0; z < N3; z++){
                    for(x = 0; x < N4; x++){
                        for(y = 0; y < N5; y++){

                            inv_var1_sum = 0;
                            for(l = z * numZ; l < (z + 1) * numZ; l++){
                                for(m = x * numXY; m < (x + 1) * numXY; m++){
                                    for(n = y * numXY; n < (y + 1) * numXY; n++){
                                        idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * z + N5 * x + y;
                                        idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + n5 * m + n;
                                        if (nhit_h[idx_h] > 0){
                                            inv_var1        = 1.0 / (rms_h[idx_h] * rms_h[idx_h]);
                                            map_l[idx_l]     += map_h[idx_h] * inv_var1;
                                            nhit_l[idx_l]    += nhit_h[idx_h];
                                        }
                                        else{
                                            inv_var1    = 0;
                                        }
                                        inv_var1_sum   += inv_var1;
                                    }
                                }
                            }
                            if (nhit_l[idx_l] > 0){
                                map_l[idx_l] /= inv_var1_sum;
                                rms_l[idx_l] = 1 / sqrt(inv_var1_sum);      
                            }
                            else{
                                map_l[idx_l] = 0;
                                rms_l[idx_l] = 0;
                            }
                        }        
                    }
                }
            }
        }
    }   
}




// Upgrade functions on pixel level
void ugradeXY4D(float* map_h, int* nhit_h, float* rms_h, 
                float* map_l, int* nhit_l, float* rms_l,
                int n0,       int n1,      int n2, 
                int n3,       int N2,      int N3,       
                int num){
    /*
    Function looping through 4D datasets of one input file and transforms the dataset from
    a low-resolution pixel grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                    
    long int p_h = N2 * N3;
    long int p_h1 = p_h * n1;
    long int idx_h;
    
    long int p_l = n2 * n3;
    long int p_l1 = p_l * n1;
    long int idx_l;

    long int i, j, k ,l, a, b;
    
    // Looping through datasets to transform each pixel into a grid of sub-pixels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(a = 0; a < n2; a++){
                for(b = 0; b < n3; b++){
                    idx_l = p_l1 * i + p_l * j + n3 * a + b;
                    
                    for(k = a * num; k < (a + 1) * num; k++){
                        for(l = b * num; l < (b + 1) * num; l++){
                            idx_h           = p_h1 * i + p_h * j + N3 * k + l;
                            map_h[idx_h]    = map_l[idx_l];       
                            nhit_h[idx_h]   = nhit_l[idx_l];       
                            rms_h[idx_h]    = rms_l[idx_l];             
                        }
                    }
                }      
            }
        }
    }
}

void ugradeXY5D(float* map_h, int* nhit_h, float* rms_h, 
                float* map_l, int* nhit_l, float* rms_l,
                int n0, int n1, int n2,
                int n3, int n4, int N3,
                int N4, int num){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a low-resolution pixel grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies    
    long int p_h = N3 * N4;
    long int p_h1 = p_h * n2;
    long int p_h2 =  p_h1 * n1;
    long int idx_h;
    
    long int p_l = n3 * n4;
    long int p_l1 = p_l * n2;
    long int p_l2 =  p_l1 * n1;
    long int idx_l;

    long int i, j, k ,l, m, a, b;

    // Looping through datasets to transform each pixel into a grid of sub-pixels.
    for(i=0; i < n0; i++){
        for(j = 0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(a = 0; a < n3; a++){
                    for(b = 0; b < n4; b++){                            
                        idx_l = p_l2 * i + p_l1 * j + p_l * k + n4 * a + b;
                        
                        for(l = a * num; l < (a + 1) * num; l++){
                            for(m = b * num; m < (b + 1) * num; m++){
                                idx_h = p_h2 * i + p_h1 * j + p_h * k + N4 * l + m;
                                map_h[idx_h]    = map_l[idx_l];       
                                nhit_h[idx_h]   = nhit_l[idx_l];       
                                rms_h[idx_h]    = rms_l[idx_l];              
                            }
                        }
                    }      
                }
            }
        }
    }
}

void ugradeXY6D(float* map_h, int* nhit_h, float* rms_h, 
                float* map_l,  int* nhit_l,  float* rms_l,
                int n0,      int n1,     int n2, 
                int n3,      int n4,     int n5,
                int N4,      int N5,     int num){
    
    /*
    Function looping through 6D datasets of one input file and transforms the dataset from
    a low-resolution pixel grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies 
    long int p_h = N4 * N5;
    long int p_h1 = p_h * n3;
    long int p_h2 =  p_h1 * n2;
    long int p_h3 =  p_h2 * n1;
    long int idx_h;

    long int p_l = n4 * n5;
    long int p_l1 = p_l * n3;
    long int p_l2 = p_l1 * n2;
    long int p_l3 = p_l2 * n1;
    long int idx_l;

    long int i, j, k ,l, m, n, a, b;
    // Looping through datasets to transform each pixel into a grid of sub-pixels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(l=0; l < n3; l++){
                    for(a = 0; a < n4; a++){
                        for(b = 0; b < n5; b++){
                            
                            idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * l + n5 * a + b;

                            for(m = a * num; m < (a + 1) * num; m++){
                                for(n = b * num; n < (b + 1) * num; n++){
                                    idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + N5 * m + n;
                                    
                                    map_h[idx_h]    = map_l[idx_l];       
                                    nhit_h[idx_h]   = nhit_l[idx_l];       
                                    rms_h[idx_h]    = rms_l[idx_l];  
                                }
                            }
                        }      
                    }
                }
            }
        }
    }
}

// Upgrading functions on frequency channel level
void ugradeZ4D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l, float* rms_l,
               int n0,       int n1,      int n2, 
               int n3,       int N1,      int num){
    /*
    Function looping through 4D datasets of one input file and transforms the dataset from
    a low-resolution frequency grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies
    long int p_h = n2 * n3;
    long int p_h1 = p_h * N1;
    long int idx_h;
    
    long int p_l = n2 * n3;
    long int p_l1 = p_l * n1;
    long int idx_l;

    long int i, j, k ,l, a;
    
    // Looping through datasets to transform each frequency into a number of sub-channels.
    for(i=0; i < n0; i++){
        for(a = 0; a < n1; a++){
            for(k = 0; k < n2; k++){
                for(l = 0; l < n3; l++){
                    for(j = a * num; j < (a + 1) * num; j++){    
                        idx_l = p_l1 * i + p_l * a + n3 * k + l;
                        idx_h = p_h1 * i + p_h * j + n3 * k + l;
                        map_h[idx_h]    = map_l[idx_l];       
                        nhit_h[idx_h]   = nhit_l[idx_l];       
                        rms_h[idx_h]    = rms_l[idx_l];       
                    }
                }
            }
        }
    }
}

void ugradeZ5D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l, float* rms_l,
               int n0, int n1, int n2,
               int n3, int n4, int N2,
               int num){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a low-resolution frequency grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies
    long int p_h = n3 * n4;
    long int p_h1 = p_h * N2;
    long int p_h2 =  p_h1 * n1;
    long int idx_h;
    
    long int p_l = n3 * n4;
    long int p_l1 = p_l * n2;
    long int p_l2 =  p_l1 * n1;
    long int idx_l;

    long int i, j, k ,l, m, a;

    // Looping through datasets to transform each frequency into a number of sub-channels.
    for(i=0; i < n0; i++){
        for(j = 0; j < n1; j++){
            for(a = 0; a < n2; a++){
                for(l = 0; l < n3; l++){
                    for(m = 0; m < n4; m++){
                        for(k = a * num; k < (a + 1) * num; k++){    
                            idx_l = p_l2 * i + p_l1 * j + p_l * a + n4 * l + m;
                            idx_h = p_h2 * i + p_h1 * j + p_h * k + n4 * l + m;
                            map_h[idx_h]    = map_l[idx_l];       
                            nhit_h[idx_h]   = nhit_l[idx_l];       
                            rms_h[idx_h]    = rms_l[idx_l];       
                        }
                    }     
                }
            }
        }
    }
}

void ugradeZ6D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l,  float* rms_l,
               int n0,       int n1,     int n2, 
               int n3,       int n4,     int n5,
               int N3,       int num){
    /*
    Function looping through 6D datasets of one input file and transforms the dataset from
    a low-resolution frequency grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel axes.
    num: int
        Number of channels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies               
    long int p_h = n4 * n5;
    long int p_h1 = p_h * N3;
    long int p_h2 =  p_h1 * n2;
    long int p_h3 =  p_h2 * n1;
    long int idx_h;

    long int p_l = n4 * n5;
    long int p_l1 = p_l * n3;
    long int p_l2 = p_l1 * n2;
    long int p_l3 = p_l2 * n1;
    long int idx_l;

    long int i, j, k ,l, m, n, a;

    // Looping through datasets to transform each frequency into a number of sub-channels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(a = 0; a < n3; a++){
                    for(m = 0; m < n4; m++){
                        for(n = 0; n < n5; n++){
                            for(l = a * num; l < (a + 1) * num; l++){
                                idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * a + n5 * m + n;
                                idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + n5 * m + n;
                                map_h[idx_h]    = map_l[idx_l];       
                                nhit_h[idx_h]   = nhit_l[idx_l];       
                                rms_h[idx_h]    = rms_l[idx_l];       
                            }
                        }
                    }
                }
            }
        }
    }
}

// Upgrading functions on pixel and frequency channel level
void ugradeXYZ4D(float* map_h, int* nhit_h, float* rms_h, 
                 float* map_l, int* nhit_l, float* rms_l,
                 int n0,       int n1,      int n2, 
                 int n3,       int N1,      int N2,       
                 int N3,       int numZ,    int numXY){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a low-resolution pixel-frequency grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel and frequency axes.
    numZ: int
        Number of channels to split into along each dimension.
    numXY: int
        Number of pixels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                
    long int p_h = N2 * N3;
    long int p_h1 = p_h * N1;
    long int idx_h;
    
    long int p_l = n2 * n3;
    long int p_l1 = p_l * n1;
    long int idx_l;

    long int i, j, k ,l, x, y, z;

    // Looping through datasets to transform each pixel-frequency cube into a number of sub-cubes.
    for(i=0; i < n0; i++){
        for(z = 0; z < n1; z++){
            for(x = 0; x < n2; x++){
                for(y = 0; y < n3; y++){
                    for(j = z * numZ; j < (z + 1) * numZ; j++){
                        for(k = x * numXY; k < (x + 1) * numXY; k++){
                            for(l = y * numXY; l < (y + 1) * numXY; l++){
                                idx_l = p_l1 * i + p_l * z + n3 * x + y;
                                idx_h = p_h1 * i + p_h * j + N3 * k + l;
                                map_h[idx_h]    = map_l[idx_l];       
                                nhit_h[idx_h]   = nhit_l[idx_l];       
                                rms_h[idx_h]    = rms_l[idx_l];       
                            }
                        }
                    }
                }        
            }
        }
    }   
}

void ugradeXYZ5D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l,  float* rms_l,    
               int n0,       int n1,            int n2,         
               int n3,       int n4,            int N2,         
               int N3,       int N4,            int numZ,       
               int numXY){

    long int p_h = N3 * N4;
    long int p_h1 = p_h * N2;
    long int p_h2 =  p_h1 * n1;
    long int idx_h;
    
    long int p_l = n3 * n4;
    long int p_l1 = p_l * n2;
    long int p_l2 =  p_l1 * n1;
    long int idx_l;

    long int i, j, k ,l, m, x, y, z;
    
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(z = 0; z < n2; z++){
                for(x = 0; x < n3; x++){
                    for(y = 0; y < n4; y++){
                        for(k = z * numZ; k < (z + 1) * numZ; k++){
                            for(l = x * numXY; l < (x + 1) * numXY; l++){
                                for(m = y * numXY; m < (y + 1) * numXY; m++){
                                    idx_l = p_l2 * i + p_l1 * j + p_l * z + n4 * x + y;
                                    idx_h = p_h2 * i + p_h1 * j + p_h * k + N4 * l + m;
                                    map_h[idx_h]    = map_l[idx_l];       
                                    nhit_h[idx_h]   = nhit_l[idx_l];       
                                    rms_h[idx_h]    = rms_l[idx_l];        
                                }
                            }
                        }
                    }        
                }
            }
        }
    }   
}

void ugradeXYZ6D(float* map_h, int* nhit_h, float* rms_h, 
               float* map_l, int* nhit_l,  float* rms_l,    
               int n0,       int n1,       int n2,         
               int n3,       int n4,       int n5,         
               int N3,       int N4,       int N5,         
               int numZ,     int numXY){
    /*
    Function looping through 5D datasets of one input file and transforms the dataset from
    a low-resolution pixel-frequency grid to a high-resolution one, to return the 
    product by call-by-pointer.

    Parameters
    ------------------------------
    map_h: float*
        Output map high-resolution dataset.
    nhit_h: int*
        Output nhit high-resolution dataset.
    rms_h: float*
        Output rms high-resolution dataset.
    map_l: float*
        Input map low-resolution dataset.
    nhit_l: int*
        Input nhit low-resolution dataset.
    rms_l: float*
        Input rms low-resolution dataset.
    ni: int
        Axis length of each dimension of the input data hyper-cubes.
    Ni: int
        Axis length of each dimension of the new length of pixel and frequency axes.
    numZ: int
        Number of channels to split into along each dimension.
    numXY: int
        Number of pixels to split into along each dimension.
    
    ------------------------------
    */  
    // Precomputing index factors for flattened indies                 

    long int p_h = N4 * N5;
    long int p_h1 = p_h * n3;
    long int p_h2 =  p_h1 * n2;
    long int p_h3 =  p_h2 * n1;
    long int idx_h;

    long int p_l = n4 * n5;
    long int p_l1 = p_l * n3;
    long int p_l2 = p_l1 * n2;
    long int p_l3 = p_l2 * n1;
    long int idx_l;

    long int i, j, k ,z, x, y, l, m, n;

    // Looping through datasets to transform each pixel into a grid of sub-pixels.
    for(i=0; i < n0; i++){
        for(j=0; j < n1; j++){
            for(k=0; k < n2; k++){
                for(z=0; z < n3; z++){
                    for(x = 0; x < n4; x++){
                        for(y = 0; y < n5; y++){
                            

                            for(l = z * numZ; l < (z + 1) * numZ; l++){
                                for(m = x * numXY; m < (x + 1) * numXY; m++){
                                    for(n = y * numXY; n < (y + 1) * numXY; n++){
                                        idx_l = p_l3 * i + p_l2 * j + p_l1 * k + p_l * z + n5 * x + y;
                                        idx_h = p_h3 * i + p_h2 * j + p_h1 * k + p_h * l + N5 * m + n;
                                        map_h[idx_h]    = map_l[idx_l];       
                                        nhit_h[idx_h]   = nhit_l[idx_l];       
                                        rms_h[idx_h]    = rms_l[idx_l];}
                                }
                            }
                        }      
                    }
                }
            }
        }
    }
}


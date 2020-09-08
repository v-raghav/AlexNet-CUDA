#include<cuda.h>
#include<iostream>
#include<cmath>

#define TILE_WIDTH 16
#define BATCH_SIZE 1
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//Structure for Convolution Parameters
struct ConvParam {
    int batchSize;
    int outChannels;
    int outHeight;
    int outWidth;
    int inChannels;
    int inHeight;
    int inWidth;
    int kerHeight;
    int kerWidth;
    int convStride;
};



//Function Declarations
void PrintMatrix(float* A, int width);
void KernelInit(float *W, const ConvParam *Param);
void InputInit(float *I, const ConvParam *Param);
void UnifiedMemoryAllocate(float **I, float **W,float **O, const ConvParam *Param);
void ValidateOutput(float *O, FILE *fp,const ConvParam *Param);

__global__ 
void conv2d_kernel(float *I, float *O, float *W, const ConvParam *Param) {

    int outOffset,inOffset,kerOffset;
    
    float Pvalue=0.0;
    int m=threadIdx.z+ blockIdx.z*blockDim.z;
    int x=threadIdx.y+ blockIdx.y*blockDim.y;
    int y=threadIdx.x+ blockIdx.x*blockDim.x;

    int outHeight=Param->outHeight;
    int outWidth=Param->outWidth;
    int kerHeight=Param->kerHeight;
    int kerWidth=Param->kerWidth;
    int inHeight=Param->inHeight;
    int inWidth=Param->inWidth;

    int kernelSize=kerHeight*kerWidth;
    int outputSize=outHeight*outWidth;
    int inputSize=inHeight*inWidth;
    int inChannels=Param->inChannels;
    int outChannels=Param->outChannels;

    if(x<outHeight && y<outWidth && m<outChannels) {
        for (int n = 0; n < Param->batchSize; n++) {
                
                    outOffset=y + x*(outWidth) +
                              m*(outputSize) +
                              n*(outputSize)*(outChannels);

                    Pvalue  = 0.0;
                    for (int k = 0; k < inChannels; k++) {
                        for (int i = 0; i < kerHeight; i++) {
                            for (int j = 0; j < kerWidth; j++) {

                                inOffset=Param->convStride * y + j + (Param->convStride * x + i)*(inWidth) +
                                         k*(inputSize) +
                                         n*(inputSize)*(inChannels);

                                kerOffset=j + i*(kerWidth) +
                                         k*(kernelSize) +
                                         m*(kernelSize)*(inChannels);         

                                Pvalue  += ( I[inOffset] ) *  ( W[kerOffset]) ;
                                
                            }
                        }
                    }
                   
                    //Pass through activation if required here
                    //O[n][m][x][y]=Activation(O[n][m][x][y]);
                    O[outOffset]=Pvalue;
        }            
    }           
            
        
}


int main() {

    srand(0);
    //These parameters reused for every layer
    ConvParam *Param;
    gpuErrchk(cudaMallocManaged(&Param, sizeof(ConvParam)));

   
    

    ////////////////  LAYER-1 /////////////////// 
     *Param={
        BATCH_SIZE,      //batchSize    
        96,      //outChannels
        55,      //outHeight
        55,      //outWidth
        3,      //inChannels
        227,      //inHeight
        227,      //inWidth
        11,      //kerHeight
        11,      //kerWidth
        4      //convStride
    }; 
   
    float *I_1, *O_1, *kernel_1;

    //Initializations

    UnifiedMemoryAllocate(&I_1,&kernel_1,&O_1,Param);
    InputInit(I_1,Param);
    KernelInit(kernel_1,Param);
  
    
    int threadsAlongZ=4;
    dim3 blocksize(TILE_WIDTH,TILE_WIDTH,threadsAlongZ);
    dim3 gridsize(ceil( ((float)Param->outWidth) / TILE_WIDTH),ceil(((float)Param->outHeight) /TILE_WIDTH),ceil(((float)Param->outChannels)/threadsAlongZ));
   

     //Convolution
    conv2d_kernel<<<gridsize,blocksize>>>(I_1,O_1,kernel_1,Param);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    PrintMatrix(I_1, 6);
   
    FILE *fp;
    fp= fopen("output.bin","rb");
    ValidateOutput(O_1,fp,Param);
    fclose(fp);


    cudaFree(I_1);
    cudaFree(kernel_1);
    cudaFree(O_1);
    cudaFree(Param);

   

    return 0;

}

void PrintMatrix(float* A, int width) {
    int i, j;
    for (i = 0; i < width; i++) {
        for (j = 0; j < width; j++) {
            
            printf("%10.2f ",A[i*width+j]);;
        }
        std::cout<<"\n";
    } 
    std::cout<<"\n";
}
void KernelInit(float *W, const ConvParam *Param) {

    int kerOffset;
    for(int m=0; m<Param->outChannels; m++) {
        for(int k=0; k<Param->inChannels; k++) {
            for(int i=0; i<Param->kerHeight; i++) {
                for(int j=0;j<Param->kerHeight; j++) {
                    kerOffset=j + i*(Param->kerWidth) +
                                         k*(Param->kerWidth)*(Param->kerHeight) +
                                         m*(Param->kerWidth)*(Param->kerHeight)*(Param->inChannels);  
                    // if(j%2==0) 
                    //     W[kerOffset]=-1+i;
                    // else
                    //     W[kerOffset]=0+j;
                    if(j%3==0)
                    W[kerOffset]=-1.0;
                    else if(j%3==1)
                    W[kerOffset]=0;
                    else
                    W[kerOffset]=1.0;                       
                }
            }
        }
    }


}

void InputInit(float *I, const ConvParam *Param) {
    int inOffset;
    for(int m=0; m<Param->batchSize; m++) {
        for(int k=0; k<Param->inChannels; k++) {
            for(int i=0; i<Param->inHeight; i++) {
                for(int j=0;j<Param->inHeight; j++) {
                    inOffset=j + i*(Param->inWidth) +
                                         k*(Param->inWidth)*(Param->inHeight) +
                                         m*(Param->inWidth)*(Param->inHeight)*(Param->inChannels);  
                                         I[inOffset]=( ((float)rand()/RAND_MAX) *2.0-1.0); //Random float between 0 and 1.
                //    if(j%8==0)
                //    I[inOffset]=0.5+(i+j)/10;
                //    else if(j%4==0)
                //    I[inOffset]=-0.4+(i+j)/10;
                //    else if(j%2==0)
                //    I[inOffset]=0.3+(i+j)/10;
                //    else
                //    I[inOffset]=0.1+(i+j)/10;        

                                   
                }
            }
        }
    }

}




void UnifiedMemoryAllocate(float **I, float **W,float **O, const ConvParam *Param){
    int inputSize=Param->batchSize*Param->inChannels*Param->inHeight*Param->inWidth;
    gpuErrchk(cudaMallocManaged(&(*I), inputSize*sizeof(float)));
    gpuErrchk(cudaMemset(*I,0, inputSize*sizeof(float)));
    
     
    int outputSize=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    gpuErrchk(cudaMallocManaged(&(*O), outputSize*sizeof(float)));
 
    int kernelSize=Param->inChannels*Param->outChannels*Param->kerHeight*Param->kerWidth;
    gpuErrchk(cudaMallocManaged(&(*W), kernelSize*sizeof(float)));
}

void ValidateOutput(float *O, FILE *fp,const ConvParam *Param) {
    int size=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    float *data=(float*) malloc (sizeof(float)*size);
    fread((void *)data, sizeof(float), size,fp);
    float max_error=0.0;
    float value;
    for(int i=0;i<size;i++){
         value=fabs(data[i]-O[i]);
         max_error=fmax(value,max_error);
        //  if(max_error>30.0){
        //      printf("Error of %f at index %d\n",max_error,i);
        //  }
        //printf("%10.2f\t%10.2f\n", data[i],O[i]);

    }
    printf("\nHighest error  is %f\n", max_error);
    free(data);
      
}

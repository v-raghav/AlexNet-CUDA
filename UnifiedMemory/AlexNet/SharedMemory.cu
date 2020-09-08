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
struct CudaParamType
	{

		int tileHeight;		//	shMem tile height
        int tileWidth;			//	shMem tile width
        int shMemHeight;
        int shMemWidth;
        int shMemSize;
        int shMemSizeTotal;

	} ;


//Function Declarations
void PrintMatrix(float* A, int width);
void KernelInit(float *W, const ConvParam *Param);
void InputInit(float *I, const ConvParam *Param);
void UnifiedMemoryAllocate(float **I, float **W,float **O, const ConvParam *Param);
void ValidateOutput(float *O, FILE *fp,const ConvParam *Param);
void CudaParamInit( CudaParamType* cudaParams,const ConvParam* Param,int Tile_Width);

/*	Grid:	x = # tiles in 2D, y = # filters, z = Batchsize 
	Block:	x = tileWidth, y = tileHeight, z = 1 * */

__global__ void convWithGPUShMem_kernel(const float* I, const float* W, float* O,
	const ConvParam* Param, const CudaParamType* cudaParams)
{
    extern __shared__ float shMem[];
    
    int shMemHeight = cudaParams->shMemHeight ;
	int shMemWidth  = cudaParams->shMemWidth ;
	int shMemSize   = cudaParams->shMemSize;

	float* I_Tile = &shMem[0];			//	Equal sized input and weight tiles
	float* W_Tile = &shMem[shMemSize];

	int kerHeight = Param->kerHeight;
	int kerWidth = Param->kerWidth;
    int I_Row;
    int I_Col;
	int I_Idx;
	int W_Idx;
	int I_TileIdx;
	int W_TileIdx;

	int inputSize = Param->inHeight * Param->inWidth;
	int filterSize = Param->kerHeight * Param->kerWidth;
	int outputSize = Param->outHeight * Param->outWidth;

	int tilesAlongY = ceil( (float)Param->outHeight / cudaParams->tileHeight);
	int tilesAlongX = ceil( (float)Param->outWidth / cudaParams->tileWidth);
	int O_TileOffsetY = (blockIdx.x / tilesAlongY) * cudaParams->tileHeight;
	int O_TileOffsetX = (blockIdx.x % tilesAlongX) * cudaParams->tileWidth;

	float pSum = 0.0;
	int n = blockIdx.z;
	int m = blockIdx.y;
	
	
	
		for (int c = 0; c < Param->inChannels; c++)
		{
			//	Copy Weights into ShMem
			if (threadIdx.x < kerWidth && threadIdx.y < kerHeight)
			{
				W_TileIdx = threadIdx.y * kerWidth + threadIdx.x;
				W_Idx = m * Param->inChannels * filterSize + c * filterSize
					+ threadIdx.y * kerWidth + threadIdx.x;
                W_Tile[W_TileIdx] = W[W_Idx];
               // W_Tile[W_TileIdx] = 0;
			}
			__syncthreads();

			//	Copy Input into ShMem
			for (int i = threadIdx.y; i < shMemHeight; i += cudaParams->tileHeight)
			{
				for (int j = threadIdx.x; j < shMemWidth; j += cudaParams->tileWidth)
				{    
                    I_Row = (Param->convStride * O_TileOffsetY + i);
                    I_Col = (Param->convStride * O_TileOffsetX + j);
                    if( I_Row < Param->inHeight && I_Col < Param->inWidth)
                    {
                        I_TileIdx = i * shMemWidth + j;
					    I_Idx = n * Param->inChannels * inputSize + c * inputSize
						    + I_Row * Param->inWidth + I_Col;
                        I_Tile[I_TileIdx] = I[I_Idx];
                       // I_Tile[I_TileIdx] = 0;
                    }
                    
				}
			}
			__syncthreads();
            if ((O_TileOffsetY + threadIdx.y) < Param->outHeight
            && (O_TileOffsetX + threadIdx.x) < Param->outWidth) {

			    //	Accumulate multiplications
                for (int fr = 0; fr < kerHeight; fr++)
                {
                    for (int fc = 0; fc < kerWidth; fc++)
                    {
                        I_TileIdx = (threadIdx.y*Param->convStride + fr) * shMemHeight + (threadIdx.x*Param->convStride + fc);
                        W_TileIdx = fr * kerWidth + fc;
                        pSum += I_Tile[I_TileIdx] * W_Tile[W_TileIdx];
                    }
                }
               
            }    
             __syncthreads();
        }
        if ((O_TileOffsetY + threadIdx.y) < Param->outHeight
        && (O_TileOffsetX + threadIdx.x) < Param->outWidth) {

            int O_Idx = n * Param->outChannels * outputSize + m * outputSize
                + (O_TileOffsetY + threadIdx.y) * Param->outWidth
                + (O_TileOffsetX + threadIdx.x);
            O[O_Idx] = pSum;
        }   
	
}



int main() {

    srand(0);
    //These parameters reused for every layer
    ConvParam *Param;
    gpuErrchk(cudaMallocManaged(&Param, sizeof(ConvParam)));
    
    CudaParamType *CudaParam;
    gpuErrchk(cudaMallocManaged(&CudaParam, sizeof(CudaParamType)));
    

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
    CudaParamInit(CudaParam,Param,TILE_WIDTH);
         
   
    dim3 blocksize_1(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridsize_1(ceil((float)(Param->outHeight)/TILE_WIDTH)*ceil(((float)Param->outWidth)/TILE_WIDTH),Param->outChannels,Param->batchSize);


    PrintMatrix(I_1, 6);

     //Convolution
    convWithGPUShMem_kernel<<<gridsize_1,blocksize_1,CudaParam->shMemSizeTotal*sizeof(float) >>>(I_1,kernel_1,O_1,Param,CudaParam);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
   
     //Dump Output to File
     FILE *fp;
     fp= fopen("output.bin","rb");
    // //OutputDump(O_5,fp,Param_5);
     ValidateOutput(O_1,fp,Param);
     fclose(fp);

    cudaFree(I_1);
    cudaFree(kernel_1);
    cudaFree(O_1);

    cudaFree(Param);
    cudaFree(CudaParam);
    
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
   // gpuErrchk(cudaMemset(*I,0, inputSize*sizeof(float)));
    
     
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
         //printf("%10.2f\t%10.2f\n", data[i],O[i]);

    }
    printf("\nHighest error percentage is %f\n", max_error);
    free(data);
      
}
void CudaParamInit( CudaParamType* cudaParams,const ConvParam* Param,int Tile_Width) {
    
    cudaParams->tileHeight=Tile_Width;
    cudaParams->tileWidth=Tile_Width;
    cudaParams->shMemHeight=(cudaParams->tileHeight- 1)*Param->convStride + Param->kerHeight,
    cudaParams->shMemWidth= (cudaParams->tileWidth- 1)*Param->convStride + Param->kerWidth ;
    cudaParams->shMemSize=cudaParams->shMemHeight *  cudaParams->shMemWidth;
    cudaParams->shMemSizeTotal=cudaParams->shMemSize+Param->kerHeight*Param->kerWidth;

}
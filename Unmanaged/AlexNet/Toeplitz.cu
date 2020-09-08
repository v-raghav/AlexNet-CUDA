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
        int shMemMul;

	} ;


//Function Declarations
void PrintMatrix(float* A, int height,int width,int channels);
void KernelInit(float *W, const ConvParam *Param);
void InputInit(float *I, const ConvParam *Param);
void UnifiedMemoryAllocate(float **I, float **W,float **O,float **U, const ConvParam *Param);
void ValidateOutput(float *O, FILE *fp,const ConvParam *Param);
void CudaParamInit( CudaParamType* cudaParams,const ConvParam* Param,int Tile_Width);
void MemoryAllocate(float **I, float **W,float **O, const ConvParam *Param);
void CudaParamInit( CudaParamType* cudaParams,const ConvParam* Param,int Tile_Width);
void DeviceAllocate(float **I, float **W,float **O,float **U, ConvParam **Param_d,CudaParamType **CudaParam_d, const ConvParam *Param);
void HostToDevice(float *I_d, float *W_d,float *O_d, ConvParam *Param_d,CudaParamType *CudaParam_d, 
                    float *I_h, float *W_h,float *O_h, ConvParam *Param_h, CudaParamType *CudaParam_h);


__global__ void MatMul(float* M, float* N, float* O,const ConvParam *Param)
{
    float OValue = 0;
    int M_Rows= Param->outChannels ;
    int M_Cols=Param->kerHeight * Param->kerWidth * Param->inChannels;
    int N_Rows=M_Cols;
    int N_Cols=Param->outWidth * Param->outHeight;
    int O_Rows=M_Rows;
    int O_Cols=N_Cols;
    int tileWidth = blockDim.x;
    int tileHeight = blockDim.y;

    int Row = blockIdx.y*tileHeight + threadIdx.y;
    int Col = blockIdx.x*tileWidth + threadIdx.x;     

    int n=blockIdx.z; 
   
    
    extern __shared__ float shMem[];
    
    int shMemTileWidth = tileWidth;      // +1 for bank conflicts 
    int shMemTileHeight = tileHeight;
    int shMemTileSize = shMemTileHeight * shMemTileWidth;
    int Md_Idx,Nd_Idx;

    float* Md = &shMem[0];
    float* Nd = &shMem[shMemTileSize];

    for (int k = 0; k < (M_Cols + tileWidth - 1)/tileWidth; k++ )
    {

        Md_Idx = (threadIdx.y) * shMemTileWidth + threadIdx.x;
        if ( (k * tileWidth + threadIdx.x) < M_Cols && Row < M_Rows)
            Md[Md_Idx] = M[  Row * M_Cols + k * tileWidth + threadIdx.x];
        else
            Md[Md_Idx] = 0.0;

        Nd_Idx = ( threadIdx.y * shMemTileWidth + threadIdx.x );
        if (k * tileWidth + threadIdx.y < N_Rows && Col < N_Cols)
            Nd[Nd_Idx] = N[n*N_Cols*N_Rows+(k * tileWidth + threadIdx.y) * N_Cols + Col];
        else
            Nd[Nd_Idx] = 0.0;

        __syncthreads();

        for (int t = 0; t < tileWidth; t++)
        {
            Md_Idx = threadIdx.y * shMemTileWidth + t;
            Nd_Idx = t * shMemTileWidth + threadIdx.x;
            OValue += Md[Md_Idx] * Nd[Nd_Idx];
        }
        __syncthreads();
    }

    if (Row < O_Rows && Col < O_Cols)
    {
      
      
        O[n*O_Cols*O_Rows+ Row * O_Cols + Col] = OValue;
    }
  
}
__global__ void ToeplitzConversion_kernel(const float *I,const ConvParam* Param, float *U,const CudaParamType* CudaParam){

    extern __shared__ float shMem[];
    int shMemHeight = CudaParam->shMemHeight ;
	int shMemWidth = CudaParam->shMemWidth;
    
    
    float* I_Tile = &shMem[0];			//	Equal sized input and weight tiles
    
    int kerHeight = Param->kerHeight;
	int kerWidth = Param->kerWidth;
   
    int I_Idx;
    int I_TileIdx;
    int outOffset;

    int inputSize = Param->inHeight * Param->inWidth;
    int kerSize = Param->kerHeight * Param->kerWidth;
    int outHeight=Param->outHeight;
    int outWidth=  Param->outWidth;
    int outputSize =outHeight*outWidth;

    int tilesAlongY = ceil( (float)outHeight / blockDim.y);
	int tilesAlongX = ceil( (float)outWidth / blockDim.x);
	int I_TileOffsetY = (blockIdx.x / tilesAlongY) * blockDim.y;
    int I_TileOffsetX = (blockIdx.x % tilesAlongX) * blockDim.x;
    
    int I_Row;
    int I_Col;
    
	int n = blockIdx.z;
    int c = blockIdx.y; //channel

     
        
        //	Copy Input into ShMem
			for (int i = threadIdx.y; i < shMemHeight; i += blockDim.y)
			{
				for (int j = threadIdx.x; j < shMemWidth; j += blockDim.x)
				{
                    I_Row = (Param->convStride * I_TileOffsetY + i);
                    I_Col = (Param->convStride * I_TileOffsetX + j);
                    if( I_Row < Param->inHeight && I_Col < Param->inWidth)
                    {
                        I_TileIdx = i * shMemWidth + j;
                        I_Idx = n * Param->inChannels * inputSize + c * inputSize
                            + (Param->convStride * I_TileOffsetY + i) * Param->inWidth
                            + (Param->convStride * I_TileOffsetX + j);
                        I_Tile[I_TileIdx] = I[I_Idx];
                    }    
					
                    
				}
			}
            __syncthreads();

            if ((I_TileOffsetY + threadIdx.y) < Param->outHeight
            && (I_TileOffsetX + threadIdx.x) < Param->outWidth) {


                for (int fr = 0; fr < kerHeight; fr++)
                {
                    for (int fc = 0; fc < kerWidth; fc++)
                    {
                        I_TileIdx = (threadIdx.y*Param->convStride + fr) * shMemHeight + (threadIdx.x*Param->convStride + fc);
                  
                        outOffset= n*outputSize*kerSize*Param->inChannels +
                                   c*outputSize*kerSize +
                                   (fc+fr*kerWidth)*outputSize +
                                   (I_TileOffsetY + threadIdx.y) * Param->outWidth +
                                   (I_TileOffsetX + threadIdx.x);

                    U[outOffset] = I_Tile[I_TileIdx];
                        
                    }
                }
            }    

    
}

int main() {

    srand(0);
    
    

    ////////////////  LAYER-1 /////////////////// 
    
    ConvParam Param_h=
    {
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
    
    float *I_h, *O_h, *kernel_h;
    CudaParamType CudaParam_h;

    float *I_d, *O_d, *kernel_d,*U_d; //U_d is the Toeplitz Matrix
    ConvParam *Param_d;
    CudaParamType *CudaParam_d;
     

    //Initializations
   
    MemoryAllocate(&I_h,&kernel_h,&O_h, &Param_h);
    InputInit(I_h, &Param_h);
    KernelInit(kernel_h, &Param_h);
    CudaParamInit(&CudaParam_h,&Param_h,TILE_WIDTH);

    DeviceAllocate(&I_d,&kernel_d,&O_d,&U_d, &Param_d,&CudaParam_d,&Param_h);
    HostToDevice(I_d, kernel_d, O_d, Param_d,CudaParam_d, I_h, kernel_h, O_h, &Param_h,&CudaParam_h);
    
   
    //Toeplitz
    dim3 blocksize_1(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridsize_1(ceil((float)Param_h.outHeight/TILE_WIDTH)*ceil((float)Param_h.outWidth/TILE_WIDTH),Param_h.inChannels,Param_h.batchSize);
    ToeplitzConversion_kernel<<<gridsize_1,blocksize_1,CudaParam_h.shMemSize*sizeof(float) >>>(I_d,Param_d,U_d,CudaParam_d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //Matrix Multiply
    dim3 mul_block_1(TILE_WIDTH,TILE_WIDTH,1);
    dim3 mul_grid_1(ceil(((float)(Param_h.outHeight* Param_h.outWidth))/TILE_WIDTH),ceil(((float) (Param_h.outChannels))/TILE_WIDTH),Param_h.batchSize);
    MatMul<<<mul_grid_1,mul_block_1, 2 * CudaParam_h.shMemMul * sizeof(float)>>>(kernel_d,U_d,O_d,Param_d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int outputSize=Param_h.batchSize * Param_h.outChannels * Param_h.outHeight * Param_h.outWidth;
    gpuErrchk(cudaMemcpy(O_h, O_d, outputSize*sizeof(float), cudaMemcpyDeviceToHost));

    

    //  FILE *fp;
    //  fp= fopen("output.bin","rb");
    //  ValidateOutput(O_h,fp,&Param_h);
    //  fclose(fp);


    cudaFree(I_d);
    cudaFree(kernel_d);
    cudaFree(O_d);
    cudaFree(Param_d);
    cudaFree(CudaParam_d);
    
    return 0;

}

void PrintMatrix(float* A, int height,int width,int channels=1) {
    int i, j;
    for(int k=0; k<channels; k++) {

        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
            
                printf("%10.2f ",A[k*width*height+i*width+j]);;
            }
        std::cout<<"\n";
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


void UnifiedMemoryAllocate(float **I, float **W,float **O,float **U, const ConvParam *Param){
    int inputSize=Param->batchSize*Param->inChannels*Param->inHeight*Param->inWidth;
    gpuErrchk(cudaMallocManaged(&(*I), inputSize*sizeof(float)));
    gpuErrchk(cudaMemset(*I,0, inputSize*sizeof(float)));
    
     
    int outputSize=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    gpuErrchk(cudaMallocManaged(&(*O), outputSize*sizeof(float)));
 
    int kernelSize=Param->inChannels*Param->outChannels*Param->kerHeight*Param->kerWidth;
    gpuErrchk(cudaMallocManaged(&(*W), kernelSize*sizeof(float)));

    int size=Param->batchSize*Param->inChannels*Param->kerWidth*Param->kerHeight*Param->outHeight*Param->outWidth;
    gpuErrchk(cudaMallocManaged(&(*U), size*sizeof(float)));

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
void CudaParamInit( CudaParamType* cudaParams,const ConvParam* Param,int Tile_Width) {
    
    cudaParams->tileHeight=Tile_Width;
    cudaParams->tileWidth=Tile_Width;
    cudaParams->shMemHeight=(cudaParams->tileHeight- 1)*Param->convStride + Param->kerHeight,
    cudaParams->shMemWidth= (cudaParams->tileWidth- 1)*Param->convStride + Param->kerWidth ;
    cudaParams->shMemSize=cudaParams->shMemHeight *  cudaParams->shMemWidth;
    cudaParams->shMemMul=Tile_Width*Tile_Width;
   

}
void HostToDevice(float *I_d, float *W_d,float *O_d, ConvParam *Param_d,CudaParamType *CudaParam_d, 
    float *I_h, float *W_h,float *O_h, ConvParam *Param_h, CudaParamType *CudaParam_h) {

    
    int inputSize=Param_h->batchSize*Param_h->inChannels*Param_h->inHeight*Param_h->inWidth;
    gpuErrchk(cudaMemcpy(I_d, I_h, inputSize*sizeof(float), cudaMemcpyHostToDevice));
 
    int kernelSize=Param_h->inChannels*Param_h->outChannels*Param_h->kerHeight*Param_h->kerWidth;
    gpuErrchk(cudaMemcpy(W_d, W_h, kernelSize*sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(Param_d, Param_h, sizeof(ConvParam), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(CudaParam_d, CudaParam_h, sizeof(CudaParamType), cudaMemcpyHostToDevice));

}
void MemoryAllocate(float **I, float **W,float **O, const ConvParam *Param) {
    
    int inputSize=Param->batchSize*Param->inChannels*Param->inHeight*Param->inWidth;
    *I=(float *) calloc(inputSize,sizeof(float));
     
    int outputSize=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    *O= (float *) calloc(outputSize,sizeof(float));
 
    int kernelSize=Param->inChannels*Param->outChannels*Param->kerHeight*Param->kerWidth;
    *W= (float *) calloc(kernelSize,sizeof(float));
}
void DeviceAllocate(float **I, float **W,float **O, float **U ,ConvParam **Param_d,CudaParamType **CudaParam_d, const ConvParam *Param_h) {
  

    int inputSize=Param_h->batchSize*Param_h->inChannels*Param_h->inHeight*Param_h->inWidth;
    gpuErrchk(cudaMalloc((void **)&(*I), inputSize*sizeof(float)));
     
    int outputSize=Param_h->batchSize*Param_h->outChannels*Param_h->outHeight*Param_h->outWidth;
    gpuErrchk(cudaMalloc((void **)&(*O), outputSize*sizeof(float)));
 
    int kernelSize=Param_h->inChannels*Param_h->outChannels*Param_h->kerHeight*Param_h->kerWidth;
    gpuErrchk(cudaMalloc((void **)&(*W), kernelSize*sizeof(float)));

    int ToeplitzSize=Param_h->batchSize*Param_h->inChannels*Param_h->kerWidth*Param_h->kerHeight*Param_h->outHeight*Param_h->outWidth;
    gpuErrchk(cudaMalloc((void **)&(*U), ToeplitzSize*sizeof(float)));

    gpuErrchk(cudaMalloc((void **)&(*Param_d), sizeof(ConvParam)));

    gpuErrchk(cudaMalloc((void **)&(*CudaParam_d), sizeof(CudaParamType)));

}
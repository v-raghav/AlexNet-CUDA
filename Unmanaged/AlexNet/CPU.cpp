#include<iostream>
#include<cstdio>
#include<cstdlib>
#define BATCH_SIZE 1
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
void convolution_2d(float *I, float *O, float *W, float *B, const ConvParam *Param );
void PrintMatrix(float* A, int width);
void KernelInit(float *W, const ConvParam *Param);
void InputInit(float *I, const ConvParam *Param);
void BiasInit(float *bias, const ConvParam *Param);
void MemoryAllocate(float **I, float **W,float **O, float **bias, const ConvParam *Param);
void OutputDump(float *O,FILE *fp,const ConvParam *Param );
void ValidateOutput(float *O, FILE *fp,const ConvParam *Param);
void toeplitz_input(float *I, const ConvParam * Param , float *U,int n);


int main() {
    srand(0);
    
    ////////////////  LAYER-1 /////////////////// 
    ConvParam Parameters_1;
    Parameters_1={
        BATCH_SIZE,      //batchSize    
       256,      //outChannels
        27,      //outHeight
        27,      //outWidth
        96,      //inChannels
        31,      //inHeight ** This is the input size including padding**
        31,      //inWidth  ** This is the input size including padding**
        5,      //kerHeight
        5,      //kerWidth
        1      //convStride
    };
    ConvParam *Param_1;
    Param_1=&Parameters_1;

    float *I_1, *O_1, *kernel_1, *bias_1;
  
    //Initializations
    MemoryAllocate(&I_1,&kernel_1,&O_1,&bias_1,Param_1);
    InputInit(I_1,Param_1);
    KernelInit(kernel_1,Param_1);
    BiasInit(bias_1,Param_1);
                         
    //Convolution
    convolution_2d(I_1,O_1,kernel_1,bias_1,Param_1);
    
    PrintMatrix(I_1, 8);
    //PrintMatrix(kernel_1,Param_1->kerHeight);
    //PrintMatrix(O_1, 6);
    //PrintMatrix(O_1,6);
      FILE *fp;
     fp= fopen("output.bin","wb");
     OutputDump(O_1,fp,Param_1);
    // //ValidateOutput(O_5,fp,Param_5);
     fclose(fp);   
   
    
    free(O_1);
    free(I_1);
    free(kernel_1);
    free(bias_1);

   
    
    return 0;




}

void convolution_2d(float *I, float *O, float *W, float *B, const ConvParam *Param ) {

    int outOffset,inOffset,kerOffset;
    for (int n = 0; n < Param->batchSize; n++) {

        for (int m = 0; m < Param->outChannels; m++) {

            for (int x = 0; x < Param->outHeight; x++) {

                for (int y = 0; y < Param->outWidth; y++) {
                    outOffset=y + x*(Param->outWidth) +
                              m*(Param->outWidth)*(Param->outHeight) +
                              n*(Param->outWidth)*(Param->outHeight)*(Param->outChannels);

                    *(O+outOffset)  = *(B+m);
                    for (int k = 0; k < Param->inChannels; k++) {
                        for (int i = 0; i < Param->kerHeight; i++) {
                            for (int j = 0; j < Param->kerWidth; j++) {

                                inOffset=Param->convStride * y + j + (Param->convStride * x + i)*(Param->inWidth) +
                                         k*(Param->inWidth)*(Param->inHeight) +
                                         n*(Param->inWidth)*(Param->inHeight)*(Param->inChannels);

                                kerOffset=j + i*(Param->kerWidth) +
                                         k*(Param->kerWidth)*(Param->kerHeight) +
                                         m*(Param->kerWidth)*(Param->kerHeight)*(Param->inChannels);         

                                *(O+outOffset)  += ( *(I+inOffset) ) *  ( *(W+kerOffset)) ;
                                
                            }
                        }
                    }
                   
                    //Pass through activation if required here
                    //O[n][m][x][y]=Activation(O[n][m][x][y]);
                }
            }
        }
    }

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
                    //     *(W+kerOffset)=-1+i;
                    // else
                    //     *(W+kerOffset)=0+j;          
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
                        // if(j%8==0)
                        // I[inOffset]=0.5+(i+j)/10;
                        // else if(j%4==0)
                        // I[inOffset]=-0.4+(i+j)/10;
                        // else if(j%2==0)
                        // I[inOffset]=0.3+(i+j)/10;
                        // else
                        // I[inOffset]=0.1+(i+j)/10;            
                }
            }
        }
    }

}

void BiasInit(float *bias, const ConvParam *Param) {
    for(int i=0;i<Param->outChannels;i++) {
        *(bias+i)=0;
    }
}

void MemoryAllocate(float **I, float **W,float **O, float **bias, const ConvParam *Param) {
    
    int inputSize=Param->batchSize*Param->inChannels*Param->inHeight*Param->inWidth;
    *I=(float *) calloc(inputSize,sizeof(float));
     
    int outputSize=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    *O= (float *) calloc(outputSize,sizeof(float));
 
    int kernelSize=Param->inChannels*Param->outChannels*Param->kerHeight*Param->kerWidth;
    *W= (float *) calloc(kernelSize,sizeof(float));

    *bias= (float *) calloc(Param->outChannels,sizeof(float));
}

void OutputDump(float *O,FILE *fp,const ConvParam *Param) {
    int size=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;


    int a=fwrite((const void *)O, sizeof(float), size,fp);
    printf("Written  %d bytes\n",a*sizeof(float));

}

void ValidateOutput(float *O, FILE *fp,const ConvParam *Param) {
    int size=Param->batchSize*Param->outChannels*Param->outHeight*Param->outWidth;
    float *data=(float*) malloc (sizeof(float)*size);
    fread((void *)data, sizeof(float), size,fp);
    float max_error=0.0;
    float value;
    for(int i=0;i<size;i++){
         value=abs((data[i]-O[i])/O[i]);
         if( value > max_error) {
             max_error=value;
         }
        

    }
    printf("\nHighest error percentage is %f\n", max_error*100.0);
    free(data);
      
}
void toeplitz_input(float *I, const ConvParam * Param , float *U,int n) {
    
    int w_base,h_roll,w_roll,inOffset,outOffset;
    for(int k=0;k<Param->inChannels;k++) {
        w_base=k*(Param->kerHeight*Param->kerWidth);
        for (int i = 0; i < Param->kerHeight; i++) {
            for (int j = 0; j < Param->kerWidth; j++) {
                 for (int x = 0; x < Param->outHeight; x++) {
                    for (int y = 0; y < Param->outWidth; y++) {
                        h_roll=w_base+i*Param->kerHeight+j;
                        w_roll=x*Param->outWidth+y;

                        inOffset=Param->convStride * y + j + (Param->convStride * x + i)*(Param->inWidth) +
                                         k*(Param->inWidth)*(Param->inHeight) +
                                         n*(Param->inWidth)*(Param->inHeight)*(Param->inChannels);
                        outOffset=h_roll*Param->kerHeight*Param->kerWidth+w_roll;

                        U[outOffset]=I[inOffset];                
                    }
                 }
            }
        }             


    }
    
     
}

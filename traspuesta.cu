// Traspuesta de una matriz	CPU y GPU

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>

StopWatchInterface *hTimer = NULL;

int BLOCK_SIZE = 0;


#ifdef DOUBLE
typedef double element; 
#else
typedef float element; 
#endif

// Function for generating random values for a matrix
void LoadStartValuesIntoMatrixRand(element *M, unsigned int mh, unsigned int mw)
{
	unsigned int i, j;
	element *reading;

	reading = M;
	for (i=0;i<mh;i++) 
		{
			for (j=0;j<mw;j++) 
				{
				 *reading = (element)(random()%9);
				 reading++;
				}
		}
}


// Function for printing a matrix
void PrintMatrix(element *M, unsigned int mh, unsigned int mw)
{
	unsigned int i, j;
	element *reading;

	reading = M;

	for (i=0;i<mh;i++) {
		for (j=0;j<mw;j++) {
			printf("%4.1f ", *reading);
			reading++;
		}
		printf("\n");
	}
	printf("\n");
}

// Function transpose sequential (rows)
void sTransposeRow (element *Min, element *Mout, unsigned int mh, unsigned int mw)
{
	unsigned int fila, col;

	for (fila=0;fila<mh;++fila)
	 for (col=0;col<mw;++col) 
		 Mout[col*mh+fila] = Min[fila*mw+col];
}

__global__ void TransposeCol(element * d_Min , element * d_Mout, unsigned int mh, unsigned int mw, unsigned int debug){
	unsigned int id_thread = (blockIdx.x*blockDim.x + threadIdx.x); // Columna -> identificador del hilo
	unsigned int col = id_thread;
	unsigned int fila = 0;
	unsigned int in;	// Offsets para los punteros
	unsigned int out; // Offsets para los punteros
	
	if (col < mw){
	 for (fila = 0; fila < mh; fila++) { // Desde fila 0 hasta la máxima -1
		in = (fila * mw) + col;	// Offset respecto a matriz de entrada
		out = (col*mh) + fila;	// Offset respecto a matriz de salida
		// mw será la anchura (mh) de la matriz traspuesta y viceversa !!!
		d_Mout[out] = d_Min[in];
		if (debug > 1) printf("d_Mout[%d][%d] = d_Min[%d][%d]\n",fila,id_thread,id_thread,fila);
	 }
	}

}
__global__ void TransposeRow(element * d_Min , element * d_Mout, unsigned int mh, unsigned int mw, unsigned int debug){
	unsigned int id_thread = (blockIdx.x*blockDim.x + threadIdx.x); // Columna -> identificador del hilo
	unsigned int col;
	unsigned int fila = id_thread; // Fila = Identificador de hilo
	unsigned int in;	// Offsets para los punteros
	unsigned int out; // Offsets para los punteros

	if (fila < mh){
	 for (col = 0; col < mw; col++) { // Desde columna 0 hasta la máxima -1
		in = (fila * mw) + col;	// Offset respecto a matriz de entrada
		out = (col*mh) + fila;	// Offset respecto a matriz de salida
		// mh será la anchura (mw) de la matriz traspuesta y viceversa !!!
		d_Mout[out] = d_Min[in];
		if (debug > 1) printf("d_Mout[%d][%d] = d_Min[%d][%d]\n",id_thread,col,col,id_thread);
	 }
	}
}
__global__ void TransposeGM(element * d_Min , element * d_Mout, unsigned int mh, unsigned int mw, unsigned int debug){
	unsigned int x, y;

	// Thread and block index
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	d_Mout[x*mh+y] = d_Min[y*mw+x];
	
	if (debug > 1) printf("d_Mout[%d][%d] = d_Min[%d][%d]\n",y,x,x,y);
}
__global__ void TransposeSM(element * d_Min , element * d_Mout, unsigned int mh, unsigned int mw, unsigned int debug){

}

// CUDA Kernels

// Reference: copy a matrix
// A thread copy a component	-	using global memory
__global__ void CopyMat (element *Min, element *Mout, unsigned int mh, unsigned int mw, unsigned int debug)
{
	unsigned int i, j;

	// Thread and block index
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	//if ( (j*mw+i) < (mh*mw) ){
		Mout[j*mw+i] = Min[j*mw+i];
	//}
}


// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{
	unsigned int mh, mw;
	unsigned int op, debug;
	struct timeval start, stop;

	#ifdef DOUBLE
		printf("[DOBLE PRECISION]\n"); 
	#else
		printf("[SIMPLE PRECISION]\n"); 
	#endif
	
	if (argc == 5)
		{
		 op = atoi(argv[1]);
		 mh = atoi(argv[2]);
		 mw = atoi(argv[3]);
		 debug = atoi(argv[4]);
		}
	else
		{
		 printf("Sintaxis: <ejecutable> option Mheight Mwidth debug\n\n");
		 printf("option:	0	A thread copies a component of the matrix\n");
		 printf("			1	A thread computes a row of the matrix\n");
		 printf("			2	A thread computes a column of the matrix\n");
		 printf("			3	A thread computes a component of the matrix (global memory)\n");
		 printf("			4	A thread computes a component of the matrix (shared memory\n");
		 printf("			5	Sequential transpose	(rows)\n");
		 printf("\ndebug: 0 no, 1 (muestra las matrices), 2 (muestra los cambios en posiciones de las matrices)\n");	
		 exit(0);
		}

	// kernel pointer
	void (*kernel)(element *, element *, unsigned int, unsigned int, unsigned int /* debug */);

	switch (op)
		{
		 case	0: kernel = &CopyMat;
					 break; 
		 case	1: kernel = &TransposeRow;
					 break;
		 case	2: kernel = &TransposeCol;
					 break;
		 case	3: kernel = &TransposeGM;
					 break;
		 case	4: kernel = &TransposeSM;
					 break;
		}

	srandom(12345);

	float timerValue;
	double totalbytes;

	// Define matrix at host
	element *Min;
	element *Mout;

	// Pointer to matrix into device
	element *d_Min;
	element *d_Mout;

	gettimeofday(&start,0);

	// Load values into Min
	Min = (element *)malloc(mh*mw*sizeof(element));
	LoadStartValuesIntoMatrixRand(Min,mh,mw);
	if (debug) PrintMatrix(Min,mh,mw);

	Mout = (element *)malloc(mh*mw*sizeof(element));

	// Calculate Mout into the device

	// allocate device memory
	checkCudaErrors(cudaMalloc((void**) &d_Min, mh*mw*sizeof(element)));
	checkCudaErrors(cudaMalloc((void**) &d_Mout, mh*mw*sizeof(element)));

	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_Min, Min, mh*mw*sizeof(element), cudaMemcpyHostToDevice));

	// setup execution parameters
	dim3 threads, grid;
	
	switch (op)
	{
		case	0:
		case	3:
		case	4:	
			threads=dim3(mw,mh);
			grid=dim3(mw/threads.x,mh/threads.y);
			break;
		case	1:
			BLOCK_SIZE = (int)ceil(sqrt(mh));
			threads=dim3(BLOCK_SIZE*BLOCK_SIZE);
			grid=dim3((int)ceil((double)mh/threads.x));
			break;
		case	2:
			BLOCK_SIZE = (int)ceil(sqrt(mw));
			threads=dim3(BLOCK_SIZE*BLOCK_SIZE);
			grid=dim3((int)ceil((double)mw/threads.x));
			break;
	}
	printf("Block_size is <%d> <%d>\n", threads.x, threads.y);	
	printf("Grid is <%d> <%d>,\n", grid.x, grid.y);
	// timers
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// execute programs
	if ((0<=op) && (op<5))
		{
		 kernel<<< grid, threads >>>(d_Min, d_Mout, mh, mw,debug);
		 cudaThreadSynchronize();	
		 sdkStopTimer(&hTimer);

		 // copy result from device to host
		 checkCudaErrors(cudaMemcpy(Mout, d_Mout, mh*mw*sizeof(element), cudaMemcpyDeviceToHost));
		}
	else if (op==5)
			 {
				sTransposeRow(Min,Mout,mh,mw);
				sdkStopTimer(&hTimer);
			 }

	// Print its value
	//printf("\nPrinting Matrix traspuesta	%dx%d\n",mw,mh);
	if (debug > 0)
		{
		 if (op==0)
			PrintMatrix(Mout,mh,mw);		// copy
		 else
			PrintMatrix(Mout,mw,mh);		// transpose
		}

	// Free matrix
	free(Min);
	free(Mout);
	cudaFree(d_Min);
	cudaFree(d_Mout);

	gettimeofday(&stop,0);
	timerValue = sdkGetTimerValue(&hTimer);
	timerValue = timerValue / 1000;
	sdkDeleteTimer(&hTimer);
	printf("Tiempo de ejecucion del kernel (segs): %f s", timerValue);
	totalbytes = 2 * sizeof(element) * mh * mw;
	printf("	%f GBs\n",(totalbytes)/timerValue/1E9);

	timerValue = (stop.tv_sec + stop.tv_usec * 1e-6)-(start.tv_sec + start.tv_usec * 1e-6);
	printf("Tiempo de ejecución total (segs) = %.6f",timerValue);
	printf("	%f GBs\n",(totalbytes)/timerValue/1E9);

	return 0;
}

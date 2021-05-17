all:
	nvcc traspuesta.cu -o traspuesta -lineinfo --ptxas-options=-v  -arch sm_35  -L /usr/local/cuda/samples/common/lib -I /usr/local/cuda/samples/common/inc

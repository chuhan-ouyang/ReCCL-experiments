#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
  ncclComm_t comms[2];

  int nDev = 4;
  int size = 32 * 1024 * 1024;
  int devs[4] = { 0, 1, 2, 3};

  // Allocate device buffers
  float *sendbuff[nDev], *recvbuff[nDev];
  cudaStream_t s[nDev];

  // Allocate host buffers for debugging
  float *hostSendBuff[nDev], *hostRecvBuff[nDev];
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaMalloc((void**)&sendbuff[i], size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&recvbuff[i], size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s[i]));

    hostSendBuff[i] = (float*)malloc(size * sizeof(float));
    hostRecvBuff[i] = (float*)malloc(size * sizeof(float));

    // GPU 0: 1.0f, GPU 1: 2.0f, GPU 2: 3.0f, GPU 3: 4.0f
    for (int j = 0; j < size; ++j) hostSendBuff[i][j] = (float)(i + 1);

    // Copy initialized values to device memory
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSendBuff[i], size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float))); // Ensure recvbuff is cleared
  }

  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // // Verify sendbuff before NCCL operation
  // for (int i = 0; i < nDev; ++i) {
  //   CUDACHECK(cudaMemcpy(hostSendBuff[i], sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
  //   printf("GPU %d sendBuff first 10 values: ", i);
  //   for (int j = 0; j < 10; j++) printf("%.1f ", hostSendBuff[i][j]);
  //   printf("\n");
  // }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaMemcpy(hostRecvBuff[i], recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 elements of recvBuff
    printf("GPU %d recvBuff first 10 values: ", i);
    for (int j = 0; j < 10; j++) {
      printf("%.1f ", hostRecvBuff[i][j]);
    }
    printf("\n");
  }

  // Free device and host buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
    free(hostSendBuff[i]);
    free(hostRecvBuff[i]);
  }

  for (int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success\n");
  return 0;
}

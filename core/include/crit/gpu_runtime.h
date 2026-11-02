#pragma once
// The device runtime Core's GPU backend is built on: the CUDA runtime, or with CRIT_CORE_HIP the HIP runtime of
// ROCm under the CUDA runtime's names. The backend (crit/cuda.h, crit/cuda_graph.h, crit/cuda_memory.h,
// crit/cuda_launch.h) is one source for both: built with CRIT_CORE_HIP=ON it is crit::hip, and an engine that
// includes these headers in a HIP build sees cudaStream_t, cudaMemcpyAsync and the rest as their HIP equivalents.
// The table names only what Core's backend uses; an engine's own CUDA code is hipified as usual.
#if defined(CRIT_CORE_HIP)
#include <hip/hip_runtime.h>

#define CUDART_CB
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceScheduleAuto hipDeviceScheduleAuto
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceScheduleMask hipDeviceScheduleMask
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorUnknown hipErrorUnknown
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDefault hipEventDefault
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceFlags hipGetDeviceFlags
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
// CUDA 12's cudaGraphInstantiate(exec, graph, flags) is HIP's hipGraphInstantiateWithFlags.
#define cudaGraphInstantiate hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraph_t hipGraph_t
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(pointer, bytes) hipHostMalloc((pointer), (bytes), hipHostMallocDefault)
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaSetDevice hipSetDevice
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamGetDevice hipStreamGetDevice
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#else
#include <cuda_runtime_api.h>
#endif

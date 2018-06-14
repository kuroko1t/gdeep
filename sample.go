package main

// #cgo CFLAGS: -I/usr/local/cuda-9.0/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
import "fmt"
import "runtime"

type Affine struct {
	dw float64
	db float64
	w float64
	b float64
}

//type layer interface {
// 	forward
// 	backward
//}

func CUDA_CHECK (error C.cudaError_t) {
	if error != 0 {
		er_message := C.GoString(C.cudaGetErrorString(error))
		fmt.Print("ERR:",er_message,"  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file,line)
	}
}

func CUDNN_CHECK (error C.cudnnStatus_t) {
	if error != C.CUDNN_STATUS_SUCCESS {
		er_message := C.GoString(C.cudnnGetErrorString(error))
		fmt.Print("ERR:",er_message,"  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file,line)
	}
}


func Host2GPU (input []float64) (unsafe.Pointer) {
	var in_data_dev unsafe.Pointer
	size := C.size_t(len(input))
	C.cudaMalloc(&in_data_dev, size)
	C.cudaMemcpy(in_data_dev, unsafe.Pointer(&input[0]), size ,C.cudaMemcpyHostToDevice)
	return in_data_dev
}

func GPU2Host (dev_data unsafe.Pointer, hostdata []float64) () {
	CUDA_CHECK(C.cudaMemcpy(unsafe.Pointer(&hostdata[0]),
	 	dev_data,
	 	C.size_t(len(hostdata)), C.cudaMemcpyDeviceToHost))
	CUDA_CHECK(C.cudaFree(dev_data))
}

//forward (affine Affine, input float64) (float64) {
//
//}


func main() {
//	var cudaStatus C.cudaError_t
	//cudaStatus = cudaSetDevices(0)
	var n C.int = 1
	var c C.int = 2
	var h C.int = 3
	var w C.int = 3
	CUDA_CHECK(C.cudaSetDevice(0))
	in_data := []float64{
		0, 1, 0 , 0, 1, 0 , 0, 1, 0,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	out_data := []float64{
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	fmt.Println("in_data",in_data)
	fmt.Println("out_data",out_data)
	in_unsafe := unsafe.Pointer(Host2GPU(in_data))
	out_unsafe := unsafe.Pointer(Host2GPU(out_data))
	// cudnn
	var cudnnHandle C.cudnnHandle_t
	var srcTensorDesc, dstTensorDesc C.cudnnTensorDescriptor_t
	CUDNN_CHECK(C.cudnnCreate(&cudnnHandle))
	CUDNN_CHECK(C.cudnnCreateTensorDescriptor(&srcTensorDesc))
	CUDNN_CHECK(C.cudnnCreateTensorDescriptor(&dstTensorDesc))
	CUDNN_CHECK(C.cudnnSetTensor4dDescriptor(srcTensorDesc, C.CUDNN_TENSOR_NCHW,
		C.CUDNN_DATA_FLOAT, n, c, h, w))
	CUDNN_CHECK(C.cudnnSetTensor4dDescriptor(dstTensorDesc, C.CUDNN_TENSOR_NCHW,
		C.CUDNN_DATA_FLOAT, n, c, h, w))
	var alpha float64 = 100000.0
	var beta float64 = 10000000.0
	CUDNN_CHECK(C.cudnnAddTensor(cudnnHandle,
	    unsafe.Pointer(&alpha),
	    srcTensorDesc,
	    in_unsafe,
	    unsafe.Pointer(&beta),
	    dstTensorDesc,
	    out_unsafe))
	GPU2Host(out_unsafe, out_data)
	fmt.Println("out_data",out_data)
	CUDA_CHECK(C.cudaFree(in_unsafe))
}

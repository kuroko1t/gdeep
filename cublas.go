package main

// #cgo CFLAGS: -I/usr/local/cuda-9.0/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
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

func CUBLAS_CHECK (error C.cublasStatus_t) {
	if error != C.CUBLAS_STATUS_SUCCESS {
		er_message := "cublas error"
		fmt.Print("ERR:",er_message,"  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file,line)
	}
}

func Host2GPU (input []float64) (*C.float) {
	var in_data_dev *C.float
	//fmt.Println(int(unsafe.Sizeof(&input)))
	size := C.size_t(len(input) * int(unsafe.Sizeof(&input)))
	fmt.Println(size)
	C.cudaMalloc(*unsafe.Pointer(in_data_dev), size)
	C.cudaMemcpy(in_data_dev, unsafe.Pointer(&input[0]), size ,C.cudaMemcpyHostToDevice)
	return in_data_dev
}

func GPU2Host (dev_data unsafe.Pointer, hostdata []float64) () {
	fmt.Println(len(hostdata) * int(unsafe.Sizeof(&hostdata)))
	CUDA_CHECK(C.cudaMemcpy(unsafe.Pointer(&hostdata[0]),
	 	dev_data,
	 	C.size_t(len(hostdata) * int(unsafe.Sizeof(&hostdata))), C.cudaMemcpyDeviceToHost))
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
	output_dim := c * h * w
	CUDA_CHECK(C.cudaSetDevice(0))
	in_data := []float64{
		0, 1, 0 , 0, 1, 0 , 0, 1, 0,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	weight := []float64{
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	out_data := []float64{
		0, 0, 0 , 0, 0, 0, 0, 0, 0,
		0, 0, 0 , 0, 0, 0, 0, 0, 0,
	}
	fmt.Println("in_data",in_data)
	fmt.Println("out_data",out_data)
	in_unsafe := unsafe.Pointer(Host2GPU(in_data))
	w_unsafe := unsafe.Pointer(Host2GPU(weight))
	out_unsafe := unsafe.Pointer(Host2GPU(out_data))
	// cudnn
	var cublasHandle C.cublasHandle_t
	CUBLAS_CHECK(C.cublasCreate(&cublasHandle))
	var alpha C.float = 1
	var beta C.float = 1
	CUBLAS_CHECK(C.cublasSgemm(cublasHandle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
	    n,
	    output_dim,
	    c * h * w,
	    &alpha,
	    in_unsafe, n,
		w_unsafe, n,
		&beta,
	    out_unsafe, n))
	GPU2Host(out_unsafe, out_data)
	fmt.Println("out_data",out_data)
	CUDA_CHECK(C.cudaFree(in_unsafe))
	CUDA_CHECK(C.cudaFree(w_unsafe))
}

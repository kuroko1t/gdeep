package main

// #cgo CFLAGS: -I/usr/local/cuda-9.0/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
import "fmt"

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

func cUDA_CHECK (C.cudaError_t error) () {
	fmt.Printf("%s",C.cudaGetErrorString(error))
}

func Host2GPU (input []float64) (unsafe.Pointer) {
	var in_data_dev unsafe.Pointer
	size := C.size_t(len(input))
	in_unsafe := unsafe.Pointer(in_data_dev)
	C.cudaMalloc(&in_unsafe, size)
	C.cudaMemcpy(unsafe.Pointer(in_data_dev), unsafe.Pointer(&input[0]), size ,C.cudaMemcpyHostToDevice)
	return in_data_dev
}

func GPU2Host (dev_data unsafe.Pointer, hostdata []float64) () {
	//var in_data_dev *float64
	//fmt.Print(len(input))
	//C.cudaMalloc(&in_unsafe, size)
	C.cudaMemcpy(unsafe.Pointer(&hostdata[0]),
	 	dev_data,
	 	C.size_t(len(hostdata)), C.cudaMemcpyDeviceToHost)
	C.cudaFree(dev_data)
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
	C.cudaSetDevice(0)
	//var in_data_dev *float64
	//	var out_data_dev *float64
	//in_unsafe := unsafe.Pointer(in_data_dev)
	//out_unsafe := unsafe.Pointer(out_data_dev)
	//in_data := [3]float64{1,1,1}
	in_data := []float64{
		0, 1, 0 , 0, 1, 0 , 0, 1, 0,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	out_data := []float64{
		0, 1, 0 , 0, 1, 0 , 0, 1, 0,
		2, 1, 1 , 1, 1, 1 , 1, 1, 1,
	}
	//fmt.Print(cap(in_data))
	//size_in := C.size_t(unsafe.Sizeof(in_data))
	//C.cudaMalloc(&in_unsafe, size_in)
	//C.cudaMalloc(&out_unsafe, size_in)
	//fmt.Print(out_data)
	//C.cudaMemcpy(unsafe.Pointer(in_unsafe), unsafe.Pointer(&in_data), size_in,C.cudaMemcpyHostToDevice)
	//C.cudaMemcpy(unsafe.Pointer(out_unsafe), unsafe.Pointer(&out_data), size_in,C.cudaMemcpyHostToDevice)
	//fmt.Print(&in_data,"\n")
	in_unsafe := unsafe.Pointer(Host2GPU(in_data))
	out_unsafe := unsafe.Pointer(Host2GPU(out_data))
	// cudnn
	var cudnnHandle C.cudnnHandle_t
	var srcTensorDesc, dstTensorDesc C.cudnnTensorDescriptor_t
	C.cudnnCreate(&cudnnHandle)
	C.cudnnCreateTensorDescriptor(&srcTensorDesc)
	C.cudnnCreateTensorDescriptor(&dstTensorDesc)

	C.cudnnSetTensor4dDescriptor(srcTensorDesc, C.CUDNN_TENSOR_NCHW, C.CUDNN_DATA_FLOAT, n, c, h, w)
	C.cudnnSetTensor4dDescriptor(dstTensorDesc, C.CUDNN_TENSOR_NCHW, C.CUDNN_DATA_FLOAT, n, c, h, w)

	var alpha float64 = 1.0
	var beta float64 = 3.0
	C.cudnnAddTensor(cudnnHandle,
	    unsafe.Pointer(&alpha),
	    srcTensorDesc,
	    in_unsafe,
	    unsafe.Pointer(&beta),
	    dstTensorDesc,
	    out_unsafe)

	//C.cudaMemcpy(unsafe.Pointer(&out_data), unsafe.Pointer(out_unsafe), size_in,C.cudaMemcpyDeviceToHost)
	fmt.Print(out_data,"\n")
	GPU2Host(out_unsafe, out_data)
	fmt.Print(out_data,"\n")
	C.cudaFree(in_unsafe)
	//C.cudaFree(out_unsafe)
}

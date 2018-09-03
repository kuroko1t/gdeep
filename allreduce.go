package gdeep

import (
	//"fmt"
	"github.com/kuroko1t/gmpi"
	"github.com/kuroko1t/gmat"
	"C"
)

func InitAllreduce() {
	mpi.Init()
}

func Allreduce(layer []LayerInterface) {
	for _, v := range layer {
		v.allreduce()
	}
	return
}

func Comm_rank() int {
	rank := mpi.Comm_rank(mpi.COMM_WORLD)
	return rank
}

func Comm_size() int {
	size := mpi.Comm_size(mpi.COMM_WORLD)
	return size
}

func performReduce(x gmat.Tensor) gmat.Tensor {
	n, c := gmat.Shape2D(x)
	x1D := gmat.Reshape2D1D(x)
	allreduceSum := make([]float64, n*c)
	mpi.Barrier(mpi.COMM_WORLD)
	mpi.Allreduce(&x1D[0], &allreduceSum[0], n*c,
		mpi.Float64, mpi.SUM, mpi.COMM_WORLD)
	commsize := mpi.Comm_size(mpi.COMM_WORLD)
	allreduceValue := make([]float64, n*c)
	for i := range allreduceSum {
		allreduceValue[i] = allreduceSum[i] / float64(commsize)
	}
	return gmat.Reshape1D2D(allreduceValue, n, c)
}

func (dense *Dense) allreduce()  {
	dense.Dw = performReduce(dense.Dw)
	dense.Db = performReduce(dense.Db)
	return
}

func (relu *Relu) allreduce() {
	return
}

func (drop *Dropout) allreduce() {
	return
}

func (softmaxWithLoss *SoftmaxWithLoss) allreduce() {
	return
}

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


func (dense *Dense) allreduce()  {
	n, c := gmat.Shape2D(dense.W)
	denseW1D := gmat.Reshape2D1D(dense.W)
	allreduceSum := make([]float64, n*c)
	mpi.Barrier(mpi.COMM_WORLD)
	mpi.Allreduce(&denseW1D[0], &allreduceSum[0], n*c,
		mpi.Float64, mpi.SUM, mpi.COMM_WORLD)
	commsize := mpi.Comm_size(mpi.COMM_WORLD)
	allreduceValue := make([]float64, n*c)
	for i := range allreduceSum {
		allreduceValue[i] = allreduceSum[i] / float64(commsize)
	}
	dense.W = gmat.Reshape1D2D(allreduceValue, n, c)
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

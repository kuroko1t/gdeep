// +build !mpi

package gdeep

func Allreduce(layer []LayerInterface) {
	return
}

func (dense *Dense) allreduce() {
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

package main

import (
	"fmt"
	"github.com/petar/GoMNIST"
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gdeep/layer"
	"github.com/kuroko1t/gmat"
)

func main() {
	batchSize := 100
	inputSize := 784
	hiddenSize := 50
	outputSize := 10
	learningRate := 0.01

	w0 := gmat.RandomArray(inputSize, hiddenSize, 0.01)
	b0 := gmat.RandomArray(batchSize, hiddenSize, 0.0)
	w1 := gmat.RandomArray(hiddenSize, outputSize, 0.01)
	b1 := gmat.RandomArray(batchSize, outputSize, 0.0)

	layer := []gdeep.LayerInterface{}
	affine1 := &gdeep.Affine{W:w0, B:b0}
	relu1 := &gdeep.Relu{}
	affine2 := &gdeep.Affine{W:w1, B:b1}
	layer = append(layer, affine1)
	layer = append(layer, relu1)
	layer = append(layer, affine2)

	//sgd := &gdeep.SGD{learningRate}
	momentum := &gdeep.Momentum{learningRate,0.9}

	train, _, _ := GoMNIST.Load("./data")
	sweeper := train.Sweep()
	for i :=0 ; i < 2000 ; i ++ {
		fmt.Println("[iteration:",i,"]")
		x, t, present:= gdeep.MnistBatch(&sweeper, batchSize)
		if !present {
			sweeper = train.Sweep()
			x, t, present = gdeep.MnistBatch(&sweeper, batchSize)
		}
		x = gdeep.ForwardLayer(layer, x)
		softmaxWithLoss := gdeep.SoftmaxWithLoss{}
		loss := softmaxWithLoss.Forward(x, t)
		common.DensePrint(loss, "loss");
		dout := gmat.MakeInit(batchSize, outputSize, 1.0)
		dout = softmaxWithLoss.Backward(dout)
		dout = gdeep.BackLayer(layer,dout)
		//gdeep.SGDUpdateLayer(layer, sgd)
		gdeep.MomentumUpdateLayer(layer, momentum)
	}
}

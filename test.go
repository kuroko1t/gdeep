package main

import (
	"fmt"
	"github.com/petar/GoMNIST"
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gdeep/layer"
	"github.com/kuroko1t/gmat"
)

func main() {
	batchSize := 2
	inputSize := 784
	hiddenSize := 50
	outputSize := 10
	learningRate := 0.1

	w0 := gmat.RandomArray(inputSize, hiddenSize, 0.01)
	b0 := gmat.RandomArray(batchSize, hiddenSize, 0.0)
	w1 := gmat.RandomArray(hiddenSize, outputSize, 0.01)
	b1 := gmat.RandomArray(batchSize, outputSize, 0.0)

	common.DenseCheck(w0,"w0")
	common.DenseCheck(b0,"b0")
	common.DenseCheck(w1,"w1")
	common.DenseCheck(b1,"b1")
	layer := []gdeep.LayerInterface{}
	affine1 := &gdeep.Affine{w0, b0, w0, w0, b0}
	relu1 := &gdeep.Relu{b0}
	affine2 := &gdeep.Affine{w1, b1, w1, w1, b1}
	layer = append(layer, affine1)
	layer = append(layer, relu1)
	layer = append(layer, affine2)

	//sgd := &gdeep.SGD{learningRate}

	train, _, _ := GoMNIST.Load("./data")
	sweeper := train.Sweep()
	for i :=0 ; i < 1 ; i ++ {
		fmt.Println("[iteration:",i,"]")
		x, t, present:= gdeep.MnistBatch(&sweeper, batchSize)
		fmt.Println(t)
		if !present {
			sweeper = train.Sweep()
			x, t, present = gdeep.MnistBatch(&sweeper, batchSize)
		}
		common.DenseCheck(x, "x");
		x = gdeep.ForwardLayer(layer, x)
		softmaxWithLoss := gdeep.SoftmaxWithLoss{}
		loss := softmaxWithLoss.Forward(x, t)
		common.DensePrint(loss, "loss");
		dout := gmat.MakeInit(batchSize, outputSize, 1.0)
		dout =	softmaxWithLoss.Backward(dout)
		dout = gdeep.BackLayer(layer,dout)
		//gdeep.SGDUpdateLayer(layer, sgd)
		common.DenseCheck(affine1.W,"affine1.w")
		common.DenseCheck(affine1.B,"affine1.b")
		common.DenseCheck(affine2.W,"affine2.w")
		common.DenseCheck(affine2.B,"affine2.b")
		common.DenseCheck(affine1.Dw,"affine1.dw")
		common.DenseCheck(affine1.Db,"affine1.db")
		common.DenseCheck(affine2.Dw,"affine2.dw")
		common.DenseCheck(affine2.Db,"affine2.db")
		affine1.W = gmat.Sub(affine1.W, gmat.MulE(affine1.Dw, learningRate))
		affine1.B = gmat.Sub(affine1.B, gmat.MulE(affine1.Db ,learningRate))
		affine2.W = gmat.Sub(affine2.W, gmat.MulE(affine2.Dw ,learningRate))
		affine2.B = gmat.Sub(affine2.B, gmat.MulE(affine2.Db ,learningRate))
	}
}

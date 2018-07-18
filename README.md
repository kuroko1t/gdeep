# gdeep
deep learning library. still in production..

# Sample

* mlp sample

```golang
package main

import (
	"fmt"
	"github.com/kuroko1t/GoMNIST"
	"github.com/kuroko1t/gdeep"
	"github.com/kuroko1t/gmat"
)

func main() {
	batchSize := 100
	inputSize := 784
	hiddenSize := 50
	outputSize := 10
	learningRate := 0.01
	iterationNum := 2000

	w0 := gmat.HeNorm2D(inputSize, hiddenSize)
	b0 := gmat.Make(batchSize, hiddenSize)
	w1 := gmat.HeNorm2D(hiddenSize, outputSize)
	b1 := gmat.Make(batchSize, outputSize)

	layer := []gdeep.LayerInterface{}
	dense1 := &gdeep.Dense{W: w0, B: b0}
	relu1 := &gdeep.Relu{}
	dense2 := &gdeep.Dense{W: w1, B: b1}
	softmaxWithLoss := gdeep.SoftmaxWithLoss{}
	layer = append(layer, dense1)
	layer = append(layer, relu1)
	layer = append(layer, dense2)

	momentum := &gdeep.Momentum{learningRate, 0.9}

	train, test, _ := GoMNIST.Load("./data")
	trainDataSize := len(train.ImagesFloatNorm)
	testDataSize := len(test.ImagesFloatNorm)
	iter := 0
	for i := 0; i < iterationNum; i++ {
		if (i+2)*batchSize > trainDataSize {
			iter = 0
		}
		x := train.ImagesFloatNorm[:][iter*batchSize : (iter+1)*batchSize]
		t := train.LabelsOneHot[:][iter*batchSize : (iter+1)*batchSize]
		x = gdeep.ForwardLayer(layer, x)
		loss := softmaxWithLoss.Forward(x, t)
		dout := gmat.MakeInit(batchSize, outputSize, 1.0)
		dout = softmaxWithLoss.Backward(dout)
		dout = gdeep.BackLayer(layer, dout)
		gdeep.MomentumUpdateLayer(layer, momentum)
		fmt.Println("[iteration:", i, "]")
		gdeep.AvePrint(loss, "loss")
		iter++
	}
	// test accuracy
	accuracy := 0.0
	iterBach := testDataSize / batchSize
	for i := 0; i < iterBach; i++ {
		x := test.ImagesFloatNorm[:][i*batchSize : (i+1)*batchSize]
		t := test.LabelsOneHot[:][i*batchSize : (i+1)*batchSize]
		x = gdeep.ForwardLayer(layer, x)
		accuracy += gdeep.Accuracy(x, t)
	}
	accuracy = accuracy / float64(iterBach)
	fmt.Println("accuracy:", accuracy)
}
```

![accuracy](https://github.com/kuroko1t/gdeep/tree/master/img/accuracy.png)

# License

gdeep is licensed under the Apache License, Version2.0

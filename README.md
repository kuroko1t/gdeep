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

	w0 := gmat.RandomNorm2D(inputSize, hiddenSize, 0.01)
	b0 := gmat.RandomNorm2D(batchSize, hiddenSize, 0.0)
	w1 := gmat.RandomNorm2D(hiddenSize, outputSize, 0.01)
	b1 := gmat.RandomNorm2D(batchSize, outputSize, 0.0)

	layer := []gdeep.LayerInterface{}
	dense1 := &gdeep.Dense{W:w0, B:b0}
	relu1 := &gdeep.Relu{}
	dense2 := &gdeep.Dense{W:w1, B:b1}
	layer = append(layer, dense1)
	layer = append(layer, relu1)
	layer = append(layer, dense2)

	momentum := &gdeep.Momentum{learningRate,0.9}

	train, _, _ := GoMNIST.Load("./data")
	sweeper := train.Sweep()
	for i :=0 ; i < 100 ; i ++ {
		fmt.Println("[iteration:",i,"]")
		x, t, present:= gdeep.MnistBatch(&sweeper, batchSize)
		if !present {
			sweeper = train.Sweep()
			x, t, present = gdeep.MnistBatch(&sweeper, batchSize)
		}
		if i == 99 {
			gdeep.Saver(layer,"./sample.gob")
		}
		x = gdeep.ForwardLayer(layer, x)
		softmaxWithLoss := gdeep.SoftmaxWithLoss{}
		loss := softmaxWithLoss.Forward(x, t)
		gdeep.AvePrint(loss, "loss");
		dout := gmat.MakeInit(batchSize, outputSize, 1.0)
		dout = softmaxWithLoss.Backward(dout)
		dout = gdeep.BackLayer(layer,dout)
		gdeep.MomentumUpdateLayer(layer, momentum)
	}
}
```

# License

gdeep is licensed under the Apache License, Version2.0

# gdeep
deep learning library written by golang. wellcom to contribution!! and still in production..

# Install

```
go get github.com/kuroko1t/GoMNIST
go get github.com/kuroko1t/gmat
go get github.com/kuroko1t/gdeep
```


## distributed

* go get github.com/kuroko1t/gmpi
* install openmpi > 3.0

# Run

* cpu

```
go run example/mlpMnist.go
```

* distributed

```
mpirun -np 2 -H host1:1,host2:1 go run example/mlpMnist_allreduce.go
```

# Document
https://godoc.org/github.com/kuroko1t/gdeep

# implemented networks
only support 2D input yet

### layer
* Dense
* Dropout

### activation
* Relu
* Sigmoid

### loss
* crossentropy

### optimization
* Sgd
* Momentum

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

	train, test, _ := GoMNIST.Load("./data")
	trainDataSize := len(train.ImagesFloatNorm)
	testDataSize := len(test.ImagesFloatNorm)

	batchSize := 128
	inputSize := 784
	hiddenSize := 20
	outputSize := 10
	learningRate := 0.01
	epochNum := 1
	iterationNum := trainDataSize * epochNum / batchSize

	dropout1 := &gdeep.Dropout{}
	dropout2 := &gdeep.Dropout{}

	layer := []gdeep.LayerInterface{}
	gdeep.LayerAdd(&layer, &gdeep.Dense{}, []int{inputSize, hiddenSize})
	gdeep.LayerAdd(&layer, &gdeep.Relu{})
	gdeep.LayerAdd(&layer, dropout1, 0.2)
	gdeep.LayerAdd(&layer, &gdeep.Dense{}, []int{hiddenSize, hiddenSize})
	gdeep.LayerAdd(&layer, &gdeep.Relu{})
	gdeep.LayerAdd(&layer, dropout2, 0.2)
	gdeep.LayerAdd(&layer, &gdeep.Dense{}, []int{hiddenSize, outputSize})
	gdeep.LayerAdd(&layer, &gdeep.SoftmaxWithLoss{})
	momentum := &gdeep.Momentum{learningRate, 0.9}

	iter := 0
	for i := 0; i < iterationNum; i++ {
		if (i+2)*batchSize > trainDataSize {
			iter = 0
		}
		imageBatch := train.ImagesFloatNorm[:][iter*batchSize : (iter+1)*batchSize]
		lagelBatch := train.LabelsOneHot[:][iter*batchSize : (iter+1)*batchSize]
		x := gmat.Make2DInitArray(imageBatch)
		t := gmat.Make2DInitArray(lagelBatch)
		loss := gdeep.Run(layer, momentum, x, t)
		gdeep.AvePrint(loss, "loss")
		iter++
	}
}

```

* distributed

[sample](https://github.com/kuroko1t/gdeep/blob/master/example/mlpMnist_allreduce.go)

* Mnist Test Accuracy

![accuracy](https://raw.githubusercontent.com/kuroko1t/gdeep/master/img/accuracy.png)

# Done

* cpu calc parallelization(dot)
* learning param save and restore
* mlp with gpu

# Todo

* CNN

# License

gdeep is licensed under the Apache License, Version2.0

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

	w0 := gmat.HeNorm2D(inputSize, hiddenSize)
	b0 := gmat.Make2D(batchSize, hiddenSize)
	w1 := gmat.HeNorm2D(hiddenSize, hiddenSize)
	b1 := gmat.Make2D(batchSize, hiddenSize)
	w2 := gmat.HeNorm2D(hiddenSize, outputSize)
	b2 := gmat.Make2D(batchSize, outputSize)

	layer := []gdeep.LayerInterface{}
	dense1 := &gdeep.Dense{W: w0, B: b0}
	relu1 := &gdeep.Relu{}
	dropout1 := &gdeep.Dropout{Train: true, Ratio: 0.2}
	dense2 := &gdeep.Dense{W: w1, B: b1}
	relu2 := &gdeep.Relu{}
	dropout2 := &gdeep.Dropout{Train: true, Ratio: 0.2}
	dense3 := &gdeep.Dense{W: w2, B: b2}
	softmaxWithLoss := &gdeep.SoftmaxWithLoss{}
	layer = append(layer, dense1, relu1)
	layer = append(layer, dropout1)
	layer = append(layer, dense2, relu2)
	layer = append(layer, dropout2)
	layer = append(layer, dense3)
	layer = append(layer, softmaxWithLoss)
	//sgd := &gdeep.SGD{learningRate}
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
		loss := gdeep.Run(layer, x, t)
		//if i == 99 {
		// 	gdeep.Saver(layer,"./sample.gob")
		//}
		gdeep.MomentumUpdateLayer(layer, momentum)
		gdeep.AvePrint(loss, "loss")
		iter++
	}

	// test accuracy
	accuracy := 0.0
	iterBach := testDataSize / batchSize
	dropout1.Train = false
	dropout2.Train = false
	for i := 0; i < iterBach; i++ {
		imageBatch := test.ImagesFloatNorm[:][i*batchSize : (i+1)*batchSize]
		lagelBatch := test.LabelsOneHot[:][i*batchSize : (i+1)*batchSize]
		x := gmat.Make2DInitArray(imageBatch)
		t := gmat.Make2DInitArray(lagelBatch)
		x = gdeep.ForwardLayer(layer, x, t)
		accuracy += gdeep.Accuracy(x, t)
	}
	accuracy = accuracy / float64(iterBach)
	fmt.Println("test accuracy:", accuracy)
}

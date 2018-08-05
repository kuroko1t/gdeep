package main

import (
	"fmt"
	"github.com/kuroko1t/GoMNIST"
	"github.com/kuroko1t/gdeep"
	"github.com/kuroko1t/gmat"
	"log"
	"os"
	"runtime/pprof"
)

func main() {
	cpuprofile := "mycpu.prof"
	f, err := os.Create(cpuprofile)
	if err != nil {
		log.Fatal(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()
	train, test, _ := GoMNIST.Load("./data")
	trainDataSize := len(train.ImagesFloatNorm)
	testDataSize := len(test.ImagesFloatNorm)

	batchSize := 128
	inputSize := 784
	hiddenSize := 512
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

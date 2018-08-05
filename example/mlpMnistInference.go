package main

import (
	"fmt"
	"github.com/kuroko1t/GoMNIST"
	"github.com/kuroko1t/gdeep"
	"github.com/kuroko1t/gmat"
)

func main() {
	batchSize := 256

	layer := []gdeep.LayerInterface{}
	gdeep.Restore("./sample.gob", &layer)
	_, test, _ := GoMNIST.Load("./data")
	testDataSize := len(test.ImagesFloatNorm)
	iterBach := testDataSize / batchSize

	accuracy := 0.0

	for i := 0; i < iterBach; i++ {
		imageBatch := test.ImagesFloatNorm[:][i*batchSize : (i+1)*batchSize]
		lagelBatch := test.LabelsOneHot[:][i*batchSize : (i+1)*batchSize]
		x := gmat.Make2DInitArray(imageBatch)
		t := gmat.Make2DInitArray(lagelBatch)
		x = gdeep.ForwardLayer(layer, x, t)
		accuracy += gdeep.Accuracy(x, t)
	}
	accuracy = accuracy / float64(iterBach)
	fmt.Println(accuracy)
}

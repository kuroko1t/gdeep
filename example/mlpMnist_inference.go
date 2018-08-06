package main

import (
	"fmt"
	"github.com/kuroko1t/GoMNIST"
	"github.com/kuroko1t/gdeep"
	"github.com/kuroko1t/gmat"
)

func main() {
	layer := []gdeep.LayerInterface{}
	gdeep.Restore("./sample.gob", &layer)
	_, test, _ := GoMNIST.Load("./data")
	imgs := [][]float64{}
	img := test.ImagesFloatNorm[:][100]
	label := test.Labels[:][100]
	imgs = append(imgs, img)
	x := gmat.Make2DInitArray(imgs)
	predictnum := gdeep.Predict(layer, x)
	fmt.Println("predict:", predictnum)
	fmt.Println("answer:", label)
}

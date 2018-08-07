package main

import (
	"fmt"
	"github.com/kuroko1t/gdeep"
)

func main() {
	img := gdeep.ImageRead2GrayNorm("data/5.jpg")
	layer := []gdeep.LayerInterface{}
	gdeep.Restore("./sample.gob", &layer)
	predictnum := gdeep.Predict(layer, img)
	fmt.Println("predict:", predictnum)
}

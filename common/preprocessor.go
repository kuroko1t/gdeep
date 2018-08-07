// Copyright 2018 kurosawa. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
package common

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
)

func ImageRead2GrayNorm(imgpath string) []float64 {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
	file, err := os.Open(imgpath)
	defer file.Close()
	if err != nil {
		fmt.Println("not ")
		log.Fatal("img file read Error")
	}
	img, _, err := image.Decode(file)
	if err != nil {
		log.Fatal("img file read Error")
	}
	maxX := img.Bounds().Max.X
	maxY := img.Bounds().Max.Y
	imgfloat := make([]float64, maxX*maxY)
	for j := 0; j < maxY; j++ {
		for i := 0; i < maxX; i++ {
			c := color.GrayModel.Convert(img.At(i, j))
			gray, _ := c.(color.Gray)
			imgfloat[i+j*maxX] = (255.0 - float64(gray.Y)) / 255.0
		}
	}
	return imgfloat
}

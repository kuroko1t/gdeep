// +build gpu

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
	"github.com/kuroko1t/gmat"
	"log"
)

var debug = false

func DenseCheck(x gmat.Tensor, name string) {
	if !debug {
		return
	}
	r, c := gmat.Shape2D(x)
	ave := gmat.Sum(x) / float64(r*c)
	max := gmat.Max(x)
	gmat.Max(x)
	log.Printf("##%s(%d,%d)## ave:%.4f max:%.4f", name, r, c, ave,max)
	//log.Printf("##%s(%d,%d)## ave:%.4f", name, r, c, ave)
}

func DensePrint(x gmat.Tensor, name string) {
	r, c := gmat.Shape2D(x)
	min := 10.0
	max := 0.0
	ave := 0.0
	for j := 0; j < r; j++ {
		for i := 0; i < c; i++ {
			value := x.CPU[j][i]
			ave += value
			if max < value {
				max = value
			}
			if min > value {
				min = value
			}
		}
	}
	ave = ave / float64(r*c)
	log.Printf("##%s(%d,%d)## ave:%.4f min:%.4f max:%.4f", name, r, c, ave, min, max)
}

func AvePrint(x gmat.Tensor, name string) {
	r, c := gmat.Shape2D(x)
	ave := gmat.Sum(x) / float64(r*c)
	log.Printf("##%s(%d,%d)## ave:%.4f", name, r, c, ave)
}

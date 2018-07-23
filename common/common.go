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
	min := 10.0
	max := 0.0
	ave := 0.0
	zero := 0
	for j := 0; j < r; j++ {
		for i := 0; i < c; i++ {
			if x.CPU[j][i] == 0 {
				zero += 1
			}
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
	log.Printf("##%s(%d,%d)## ave:%.4f min:%.4f max:%.4f zero:%d", name, r, c, ave, min, max, zero)
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
	log.Printf("##%s(%d,%d)## ave:%.4f", name, r, c, ave)
}

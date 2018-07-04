package common

import (
	//"fmt"
	//"math"
	//"math/rand"
    //"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
	"log"
)

var debug = true

func DenseCheck(x *mat.Dense, name string) {
	if debug {
		return
	}
	r, c := x.Dims()
	min := 10.0
	max := 0.0
	ave := 0.0
	for j := 0; j < r; j++ {
		for i := 0; i < c; i++ {
			value := x.At(j,i)
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
	log.Printf("##%s(%d,%d)## ave:%.4f min:%.4f max:%.4f",name,r,c,ave,min,max)
}

func DensePrint(x *mat.Dense, name string) {
	r, c := x.Dims()
	min := 10.0
	max := 0.0
	ave := 0.0
	for j := 0; j < r; j++ {
		for i := 0; i < c; i++ {
			value := x.At(j,i)
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
	log.Printf("##%s(%d,%d)## ave:%.4f min:%.4f max:%.4f",name,r,c,ave,min,max)
}

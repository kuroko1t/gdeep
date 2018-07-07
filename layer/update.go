package gdeep

import (
	//"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
)

type SGD struct {
	LearningRate float64
}

type UpdateInterface interface {
	Update([][]float64) [][]float64
}

func (sgd *SGD) Update(x [][]float64) [][]float64 {
	y := gmat.MulE(x, (1.0 - sgd.LearningRate))
	return y
}

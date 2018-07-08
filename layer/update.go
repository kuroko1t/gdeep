package gdeep

import (
	//"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
)

type SGD struct {
	LearningRate float64
}

func (affine *Affine) SGDUpdate(sgd *SGD) {
	affine.W = gmat.Sub(affine.W, gmat.MulE(affine.Dw, sgd.LearningRate))
	affine.B = gmat.Sub(affine.B, gmat.MulE(affine.Db, sgd.LearningRate))
	return
}

func (relu *Relu) SGDUpdate(sgd *SGD) {
	return
}

func SGDUpdateLayer(layer []LayerInterface, sgd *SGD) {
	for _, v := range layer {
		v.SGDUpdate(sgd)
	}
	return
}

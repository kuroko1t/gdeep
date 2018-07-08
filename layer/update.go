package gdeep

import (
	"github.com/kuroko1t/gmat"
	"github.com/kuroko1t/gdeep/common"
)

type SGD struct {
	LearningRate float64
}

func SGDUpdateLayer(layer []LayerInterface, sgd *SGD) {
	for _, v := range layer {
		v.SGDUpdate(sgd)
	}
	return
}

func (affine *Affine) SGDUpdate(sgd *SGD) {
	affine.W = gmat.Sub(affine.W, gmat.MulE(affine.dw, sgd.LearningRate))
	affine.B = gmat.Sub(affine.B, gmat.MulE(affine.db, sgd.LearningRate))
	return
}

func (relu *Relu) SGDUpdate(sgd *SGD) {
	return
}


type Momentum struct {
	LearningRate float64
	MomentumRate float64
}

func MomentumUpdateLayer(layer []LayerInterface, m *Momentum) {
	for _, v := range layer {
		v.MomentumUpdate(m)
	}
	return
}

func (affine *Affine)MomentumUpdate(m *Momentum) {
	//common.DenseCheck(affine.W, "momentum affine.W")
	//common.DenseCheck(affine.B, "momentum affine.B")
	affine.W = MomentumCalc(m, affine.W, affine.dw, &affine.vw)
	affine.B = MomentumCalc(m, affine.B, affine.db, &affine.vb)
	common.DenseCheck(affine.W, "momentum affine.W")
	common.DenseCheck(affine.B, "momentum affine.B")
}

func MomentumCalc(m *Momentum, layerParam [][]float64, layerDelta [][]float64, v *[][]float64) [][]float64 {
	if *v == nil {
		*v = gmat.Make(len(layerParam),len(layerParam[0]))
	}
	av := gmat.MulE(*v , m.MomentumRate)
	dl := gmat.MulE(layerDelta, m.LearningRate)
	*v = gmat.Sub(av ,dl)
	layerParam = gmat.Add(layerParam, *v)
	return layerParam
}

func (relu *Relu) MomentumUpdate(m *Momentum) {
	return
}

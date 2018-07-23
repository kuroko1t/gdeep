package layer

import (
	"github.com/kuroko1t/gmat"
)

type Sigmoid struct {
	Out gmat.Tensor
}

type Dense struct {
	W  gmat.Tensor
	B  gmat.Tensor
	X  gmat.Tensor
	Dw gmat.Tensor
	Db gmat.Tensor
	Vw gmat.Tensor
	Vb gmat.Tensor
}

type Dropout struct {
	Mask  gmat.Tensor
	Train bool
	Ratio float64
}

type SoftmaxWithLoss struct {
	Loss gmat.Tensor
	Y    gmat.Tensor
	T    gmat.Tensor
}

type Relu struct {
	Mask gmat.Tensor
}

type SGD struct {
	LearningRate float64
}

type Momentum struct {
	LearningRate float64
	MomentumRate float64
}

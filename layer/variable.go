package layer

import (
	"github.com/kuroko1t/gmat"
)

type Sigmoid struct {
	Out gmat.Data
}

type Dense struct {
	W  gmat.Data
	B  gmat.Data
	X  gmat.Data
	Dw gmat.Data
	Db gmat.Data
	Vw gmat.Data
	Vb gmat.Data
}

type Dropout struct {
	Mask  gmat.Data
	Train bool
	Ratio float64
}

type SoftmaxWithLoss struct {
	Loss gmat.Data
	Y    gmat.Data
	T    gmat.Data
}

type Relu struct {
	Mask gmat.Data
}

type SGD struct {
	LearningRate float64
}

type Momentum struct {
	LearningRate float64
	MomentumRate float64
}

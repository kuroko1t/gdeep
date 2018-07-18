package layer

type Sigmoid struct {
	Out [][]float64
}

type Dense struct {
	W  [][]float64
	B  [][]float64
	X  [][]float64
	Dw [][]float64
	Db [][]float64
	Vw [][]float64
	Vb [][]float64
}

type Dropout struct {
	Mask  [][]float64
	Train bool
	Ratio float64
}

type SoftmaxWithLoss struct {
	Loss [][]float64
	Y    [][]float64
	T    [][]float64
}

type Relu struct {
	Mask [][]float64
}

type SGD struct {
	LearningRate float64
}

type Momentum struct {
	LearningRate float64
	MomentumRate float64
}

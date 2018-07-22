package layer

import (
	"github.com/kuroko1t/gmat"
	"math"
)

func MaskFunc(v float64) float64 {
	if v <= 0 {
		v = 0
	} else {
		v = 1
	}
	return v
}

func Softmax(a gmat.Data) gmat.Data {
	a = gmat.Apply(a, Exp)
	sumExp := gmat.SumCol(a)
	sumExp = gmat.Div(a, sumExp)
	return sumExp
}

func Sigmoid_f(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func Sigmoid_b(v float64) float64 {
	return 1 - v
}

func Add1(v float64) float64 {
	return v + 1
}

func Exp(v float64) float64 {
	return float64(math.Exp(float64(v)))
}

func crossEnrtopy(v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

func CrossEnrtopyError(y , t gmat.Data) gmat.Data {
	y = gmat.Apply(y, crossEnrtopy)
	y = gmat.Mul(t, y)
	y = gmat.MulE(gmat.SumCol(y), -1)
	return y
}

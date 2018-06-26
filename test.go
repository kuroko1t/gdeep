package main

import "fmt"
import "gonum.org/v1/gonum/mat"
import "math"

type Sigmoid struct {
	out *mat.Dense
}

type Affine struct {
	w  *mat.Dense
	b  *mat.Dense
	x  *mat.Dense
	dw *mat.Dense
	db *mat.Dense
}

type SoftmaxWithLoss struct {
	loss *mat.Dense
	y    *mat.Dense
	t    *mat.Dense
}

func (sigmoid *Sigmoid) forward(x *mat.Dense) *mat.Dense {
	x.Apply(sigmoid_f, x)
	sigmoid.out = x
	return x
}

func (sigmoid *Sigmoid) backward(dout *mat.Dense) *mat.Dense {
	r, c := dout.Dims()
	dx := mat.NewDense(r, c, nil)
	fmt.Println("koko", sigmoid.out)
	dx.Apply(sigmoid_b, dout)
	fmt.Println("koko", sigmoid.out)
	dx.MulElem(dx, sigmoid.out)
	dx.MulElem(dx, dout)
	return dx
}

func sigmoid_f(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func sigmoid_b(i, j int, v float64) float64 {
	return 1 - v
}

func add1(i, j int, v float64) float64 {
	return v + 1
}

func minus(i, j int, v float64) float64 {
	return -1 * v
}

func (affine *Affine) init(w *mat.Dense, b *mat.Dense) {
	affine.w = w
	affine.b = b
}

func (affine *Affine) forward(x *mat.Dense) *mat.Dense {
	affine.x = x
	x.Mul(x, affine.w)
	x.Add(x, affine.b)
	return x
}

func (affine *Affine) backward(dout *mat.Dense) *mat.Dense {
	r, c := affine.w.Dims()
	wt := mat.NewDense(r, c, nil)
	xt := mat.NewDense(r, c, nil)
	dx := mat.NewDense(r, c, nil)
	wt.Inverse(affine.w)
	dx.Mul(dout, wt)

	xt.Inverse(affine.x)
	affine.dw.Mul(xt, dout)
	affine.db = sumRow(dout)
	return dx
}

func sumRow(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	sumArray := make([]float64, c)
	for j := 0; j < c; j++ {
		sumValue := 0.0
		for i := 0; i < r; i++ {
			sumValue += m.At(i, j)
			sumArray[i+j*r] = sumValue
		}
	}
	sums := mat.NewDense(r, c, sumArray)
	return sums
}

func sumCol(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	sumArray := make([]float64, r*c)
	for j := 0; j < r; j++ {
		sumValue := 0.0
		for i := 0; i < c; i++ {
			sumValue += m.At(j, i)
			sumArray[i+j*c] = sumValue
		}
	}
	sums := mat.NewDense(r, c, sumArray)
	return sums
}

// softmax_withloss

func (softmaxWithLoss *SoftmaxWithLoss) forward(x *mat.Dense, t *mat.Dense) *mat.Dense {
	softmaxWithLoss.t = t
	softmaxWithLoss.y = softmax(x)
	softmaxWithLoss.loss = crossEnrtopyError(softmaxWithLoss.y, softmaxWithLoss.t)
	return softmaxWithLoss.loss
}

func (softmaxWithLoss *SoftmaxWithLoss) backward() (dx *mat.Dense) {
	batchSize , c := softmaxWithLoss.t.Dims()
	batchSizeArray := mat.NewDense(batchSize, c, make([]float64, batchSize*c,batchSize))
	dx.Sub(softmaxWithLoss.y, softmaxWithLoss.t)
	dx.DivElem(dx,batchSizeArray)
	return dx
}

func exp(i, j int, v float64) float64 {
	return math.Exp(v)
}

func softmax(a *mat.Dense) *mat.Dense {
	a.Apply(exp, a)
	r, _ := a.Dims()
	sumExp := mat.NewDense(1, r, nil)
	sumExp = sumCol(a)
	sumExp.DivElem(a,sumExp)
	return sumExp
}

func crossEnrtopyError(y *mat.Dense, t *mat.Dense) *mat.Dense {
	y.Apply(crossEnrtopy,y)
	y.MulElem(t,y)
	y.Apply(minus,sumCol(y))
	return y
}

func crossEnrtopy(i, j int, v float64) float64 {
	delta := 1e-7
	return math.Log(v + delta)
}

func main() {
	zero := mat.NewDense(3, 5, nil)
	zero.Apply(add1, zero)
	//sigmoid := Sigmoid{zero}
	//sigmoid.forward(zero)
	//sigmoid.backward(zero)
	//fmt.Print(zero)
	softmax(zero)
	crossEnrtopyError(zero,zero)
}

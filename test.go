package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

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

type Relu struct {
	mask *mat.Dense
}

func (relu *Relu) forward(x *mat.Dense) *mat.Dense {
	fmt.Println(x)
	fmt.Println(relu.mask)
	relu.mask.Apply(maskFunc, x)
	x.MulElem(x, relu.mask)
	return x
}

func (relu *Relu) backward(dout *mat.Dense) *mat.Dense {
	dout.MulElem(dout, relu.mask)
	return dout
}

func maskFunc(i, j int, v float64) float64 {
	if v <= 0 {
		v = 0
	} else {
		v = 1
	}
	return v
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
	var c mat.Dense
	c.Mul(x, affine.w)
	c.Add(&c, affine.b)
	fmt.Println(c)
	return &c
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
	batchSize, c := softmaxWithLoss.t.Dims()
	batchSizeArray := mat.NewDense(batchSize, c, make([]float64, batchSize*c, batchSize))
	dx.Sub(softmaxWithLoss.y, softmaxWithLoss.t)
	dx.DivElem(dx, batchSizeArray)
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
	sumExp.DivElem(a, sumExp)
	return sumExp
}

func crossEnrtopyError(y *mat.Dense, t *mat.Dense) *mat.Dense {
	y.Apply(crossEnrtopy, y)
	y.MulElem(t, y)
	y.Apply(minus, sumCol(y))
	return y
}

func crossEnrtopy(i, j int, v float64) float64 {
	delta := 1e-7
	return math.Log(v + delta)
}

func randomArray(data []float64) []float64 {
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return data
}

type ForwardInterface interface {
	forward() *mat.Dense
}

func Update(layer []ForwardInterface) {
	for f,v := range layer {
		fmt.Println(f)
		fmt.Println(v)
	}
}

func main() {
	//zero.Apply(add1, zero)
	batchSize := 3
	inputSize := 12
	hiddenSize := 2
	outputSize := 10
	xData := make([]float64, batchSize*inputSize)
	tData := make([]float64, batchSize*outputSize)
	w0Data := make([]float64, inputSize*hiddenSize)
	b0Data := make([]float64, batchSize*hiddenSize)
	w1Data := make([]float64, hiddenSize*outputSize)
	b1Data := make([]float64, batchSize*outputSize)
	x := mat.NewDense(batchSize, inputSize, randomArray(xData))
	w0 := mat.NewDense(inputSize, hiddenSize, randomArray(w0Data))
	b0 := mat.NewDense(batchSize, hiddenSize, randomArray(b0Data))
	w1 := mat.NewDense(hiddenSize, outputSize, randomArray(w1Data))
	b1 := mat.NewDense(batchSize, outputSize, randomArray(b1Data))
	t := mat.NewDense(batchSize, outputSize, randomArray(tData))
	//w2 := mat.NewDense(4, 9, randomArray(data))
	//b2 := mat.NewDense(4, 9, randomArray(data))
	affine1 := Affine{w0, b0, w0, w0, b0}
	affine2 := Affine{w1, b1, w1, w1, b1}
	relu1 := Relu{b0}
	softmaxWithLoss := SoftmaxWithLoss{}
	//fmt.Print(x)
	layer := []ForwardInterface{}
	layer = append(layer, &Affine{w0, b0, w0, w0, b0})
	layer = append(layer, &Relu{b0})
	layer = append(layer, &Affine{w1, b1, w1, w1, b1})
	//layer.append(layer, &SoftmaxWithLoss{})
	//x = affine1.forward(x)
	//x = relu1.forward(x)
	//x = affine2.forward(x)
	//softmaxWithLoss.forward(x, t)
	fmt.Println(x)
	//sigmoid.backward(zero)
	//softmax(zero)
	//crossEnrtopyError(zero, zero)
	//fmt.Print(x)
}

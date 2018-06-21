package main

import "fmt"
import "gonum.org/v1/gonum/mat"
import "math"

type Sigmoid struct {
	out *mat.Dense
}

type Affine struct {
	w *mat.Dense
	b *mat.Dense
	x *mat.Dense
	dw *mat.Dense
	db *mat.Dense
}

type SoftmaxWithLoss struct {
	loss *mat.Dense
	y *mat.Dense
	t *mat.Dense
}

func (sigmoid *Sigmoid) forward(x *mat.Dense) (*mat.Dense) {
	x.Apply(sigmoid_f,x)
	sigmoid.out = x
	return x
}

func (sigmoid *Sigmoid) backward(dout *mat.Dense) (*mat.Dense) {
	r, c := dout.Dims()
	dx := mat.NewDense(r, c, nil)
	fmt.Println("koko",sigmoid.out)
	dx.Apply(sigmoid_b,dout)
	fmt.Println("koko",sigmoid.out)
	dx.MulElem(dx, sigmoid.out)
	dx.MulElem(dx,dout)
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

func (affine *Affine) init(w *mat.Dense, b *mat.Dense) () {
	affine.w = w
	affine.b = b
}


func (affine *Affine) forward(x *mat.Dense) (*mat.Dense) {
	affine.x = x
	x.Mul(x, affine.w)
	x.Add(x,affine.b)
	return x
}

func (affine *Affine) backward(dout *mat.Dense) (*mat.Dense) {
	r, c := affine.w.Dims()
	wt := mat.NewDense(r, c, nil)
	xt := mat.NewDense(r, c, nil)
	dx := mat.NewDense(r, c, nil)
	wt.Inverse(affine.w)
	dx.Mul(dout, wt)

	xt.Inverse(affine.x)
	affine.dw.Mul(xt, dout)
	affine.db = sum(dout)
	return dx
}

func sum(m *mat.Dense) (*mat.Dense) {
	r, c := m.Dims()
	sum_array:=make([]float64,c)
	for j:=0; j< c ; j++ {
		sum_value := 0.0
		for i:=0; i< r ; r++ {
			sum_value += m.At(i,j)
		}
		sum_array[j] = sum_value
	}
	sums := mat.NewDense(1, c, sum_array)
	return sums
}

// softmax_withloss

func (softmaxWithLoss *SoftmaxWithLoss) forward(x *mat.Dense, t *mat.Dense) (*mat.Dense) {
	softmaxWithLoss.t = t
	sigmoid.out = x
	return x
}

func (softmaxWithLoss *SoftmaxWithLoss) backward(dout *mat.Dense) (*mat.Dense) {
	r, c := dout.Dims()
	dx := mat.NewDense(r, c, nil)
	fmt.Println("koko",sigmoid.out)
	dx.Apply(sigmoid_b,dout)
	fmt.Println("koko",sigmoid.out)
	dx.MulElem(dx, sigmoid.out)
	dx.MulElem(dx,dout)
	return dx
}


func main() {
	zero := mat.NewDense(3, 5, nil)
	zero.Apply(add1,zero)
	sigmoid := Sigmoid{zero}
	sigmoid.forward(zero)
	sigmoid.backward(zero)
	//fmt.Print(zero)
}

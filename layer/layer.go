package gdeep

import (
	"math"
	"github.com/petar/GoMNIST"
	"github.com/kuroko1t/gdeep/common"
	"fmt"
	"github.com/kuroko1t/gmat"
)

type Sigmoid struct {
	Out [][]float64
}

type Affine struct {
	W  [][]float64
	B  [][]float64
	X  [][]float64
	Dw [][]float64
	Db [][]float64
}

type SoftmaxWithLoss struct {
	Loss [][]float64
	Y    [][]float64
	T    [][]float64
}

type Relu struct {
	Mask [][]float64
}

func (relu *Relu) Forward(x [][]float64) [][]float64 {
	x = gmat.Apply(x, maskFunc)
	x = gmat.Mul(x, relu.Mask)
	common.DenseCheck(x, "Relu forward output")
	return x
}

func (relu *Relu) Backward(dout [][]float64) [][]float64 {
	dout = gmat.Mul(dout, relu.Mask)
	common.DenseCheck(dout, "Relu backward output")
	return dout
}

func maskFunc(v float64) float64 {
	if v <= 0 {
		v = 0
	} else {
		v = 1
	}
	return v
}

func (sigmoid *Sigmoid) Forward(x [][]float64) [][]float64 {
	x = gmat.Apply(x, sigmoid_f)
	sigmoid.Out = x
	return x
}

func (sigmoid *Sigmoid) Backward(dout [][]float64) [][]float64 {
	dx := gmat.Apply(dout, sigmoid_b)
	dx = gmat.Mul(dx, sigmoid.Out)
	dx = gmat.Mul(dx, dout)
	return dx
}

func sigmoid_f(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func sigmoid_b( v float64) float64 {
	return 1 - v
}

func add1(v float64) float64 {
	return v + 1
}

//func (affine *Affine) init(w [][]float64, b [][]float64) {
// 	affine.W = w
// 	affine.B = b
//}

func (affine *Affine) Forward(x [][]float64) [][]float64 {
	common.DenseCheck(x, "affine forward input")
	fmt.Println("x",x)
	affine.X = x
	c := gmat.Dot(x, affine.W)
	c = gmat.Add(c, affine.B)
	common.DenseCheck(c, "affine forward output")
	return c
}

func (affine *Affine) Backward(dout [][]float64) [][]float64 {
	wt := gmat.T(affine.W)
	//fmt.Println(dout)
	common.DenseCheck(dout, "affine backward dout")
	common.DenseCheck(wt, "affine backward wt")
	dx := gmat.Dot(dout, wt)
	xt := gmat.T(affine.X)
	fmt.Println("dout",dout)
	fmt.Println("xt",xt)
	affine.Dw = gmat.Dot(xt, dout)
	common.DenseCheck(affine.Dw, "affine backward affine.Dw")
	affine.Db = gmat.SumRow(dout)
	common.DenseCheck(dx,"affine backward output")
	common.DenseCheck(affine.Dw,"affine backward dw")
	return dx
}

func (softmaxWithLoss *SoftmaxWithLoss) Forward(x [][]float64, t [][]float64) [][]float64 {
	softmaxWithLoss.T = t
	softmaxWithLoss.Y = softmax(x)
	softmaxWithLoss.Loss = crossEnrtopyError(softmaxWithLoss.Y, softmaxWithLoss.T)
	common.DenseCheck(softmaxWithLoss.Loss, "softmaxwithloss forward")
	return softmaxWithLoss.Loss
}

func (softmaxWithLoss *SoftmaxWithLoss) Backward(dout [][]float64) ([][]float64) {
	batchSize := len(softmaxWithLoss.T)
	submat := gmat.Sub(softmaxWithLoss.Y, softmaxWithLoss.T)
	submat = gmat.MulE(submat, 1/float64(batchSize))
	common.DenseCheck(submat, "softmaxwithloss backward")
	return submat
}

func exp(v float64) float64 {
	return math.Exp(v)
}

func softmax(a [][]float64) [][]float64 {
	a = gmat.Apply(a, exp)
	sumExp := gmat.SumCol(a)
	sumExp = gmat.Div(a, sumExp)
	return sumExp
}

func crossEnrtopyError(y [][]float64, t [][]float64) [][]float64 {
	y = gmat.Apply(y, crossEnrtopy)
	y = gmat.Mul(t, y)
	common.DenseCheck(y, "crossEnrtopyError before sumcol");
	y = gmat.MulE(gmat.SumCol(y), -1)
	common.DenseCheck(y, "crossEnrtopyError outout");
	return y
}

func crossEnrtopy(v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

type LayerInterface interface {
	Forward([][]float64) [][]float64
	Backward([][]float64) [][]float64
	SGDUpdate(*SGD)
}

func ForwardLayer(layer []LayerInterface, x [][]float64) [][]float64 {
	for _, v := range layer {
		x = v.Forward(x)
	}
	return x
}

func BackLayer(layer []LayerInterface, dout [][]float64) [][]float64 {
	for i := len(layer)-1; i >=0; i-- {
		f := layer[i]
		dout = f.Backward(dout)
	}
	return dout
}

func OneHot(x int, size int) (y []float64) {
	y = make([]float64, size)
	y[x] = 1
	return y
}

func MnistBatch(sweep **GoMNIST.Sweeper, batchSize int) ([][]float64 ,[][]float64, bool) {
	outputSize := 10
	imagenum := 784
	image := gmat.Make(batchSize, imagenum)
	label := gmat.Make(batchSize, outputSize)
	present := true
	for i:= 0; i < batchSize ; i++ {
		image_tmp, label_tmp, present := (*sweep).Next()
		labelTmpOneHot := OneHot(int(label_tmp), outputSize)
		if !present {
			return make([][]float64,2), make([][]float64,2), present
		}
		for o := 0 ; o < len(image_tmp) ; o++ {
			image[i][o] = float64(image_tmp[o])/255
		}
		for o := 0 ; o < len(labelTmpOneHot) ; o++ {
			label[i][o] = float64(labelTmpOneHot[o])
		}
	}
	return image, label ,present
}

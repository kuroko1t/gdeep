package gdeep

import (
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gdeep/layer"
	"github.com/kuroko1t/gmat"
	"math/rand"
)

type Relu layer.Relu
type Sigmoid layer.Sigmoid
type Dense layer.Dense
type Dropout layer.Dropout
type SoftmaxWithLoss layer.SoftmaxWithLoss
type SGD layer.SGD
type Momentum layer.Momentum

func (relu *Relu) Forward(x gmat.Tensor) gmat.Tensor {
	relu.Mask = gmat.Apply(x, layer.MaskFunc)
	x = gmat.Mul(x, relu.Mask)
	common.DenseCheck(x, "Relu forward output")
	return x
}

func (relu *Relu) Backward(dout gmat.Tensor) gmat.Tensor {
	dout = gmat.Mul(dout, relu.Mask)
	common.DenseCheck(dout, "Relu backward output")
	return dout
}

func (sigmoid *Sigmoid) Forward(x gmat.Tensor) gmat.Tensor {
	x = gmat.Apply(x, layer.Sigmoid_f)
	sigmoid.Out = x
	return x
}

func (sigmoid *Sigmoid) Backward(dout gmat.Tensor) gmat.Tensor {
	dx := gmat.Apply(dout, layer.Sigmoid_b)
	dx = gmat.Mul(dx, sigmoid.Out)
	dx = gmat.Mul(dx, dout)
	return dx
}

func (dense *Dense) Forward(x gmat.Tensor) gmat.Tensor {
	common.DenseCheck(x, "dense forward input")
	dense.X = x
	c := gmat.Dot(x, dense.W)
	c = gmat.Add(c, dense.B)
	common.DenseCheck(c, "dense forward output")
	return c
}

func (dense *Dense) Backward(dout gmat.Tensor) gmat.Tensor {
	wt := gmat.T(dense.W)
	dx := gmat.Dot(dout, wt)
	xt := gmat.T(dense.X)
	dense.Dw = gmat.Dot(xt, dout)
	dense.Db = gmat.SumRow(dout)
	common.DenseCheck(dx, "dense backward output")
	return dx
}

func (softmaxWithLoss *SoftmaxWithLoss) Forward(x gmat.Tensor, t gmat.Tensor) gmat.Tensor {
	softmaxWithLoss.T = t
	softmaxWithLoss.Y = layer.Softmax(x)
	softmaxWithLoss.Loss = layer.CrossEnrtopyError(softmaxWithLoss.Y, softmaxWithLoss.T)
	common.DenseCheck(softmaxWithLoss.Loss, "softmaxwithloss forward")
	return softmaxWithLoss.Loss
}

func (softmaxWithLoss *SoftmaxWithLoss) Backward(dout gmat.Tensor) gmat.Tensor {
	batchSize, _ := gmat.Shape2D(softmaxWithLoss.T)
	submat := gmat.Sub(softmaxWithLoss.Y, softmaxWithLoss.T)
	submat = gmat.MulE(submat, 1/float64(batchSize))
	common.DenseCheck(submat, "softmaxwithloss backward")
	return submat
}

func (drop *Dropout) Forward(x gmat.Tensor) gmat.Tensor {
	if drop.Train {
		n, m := gmat.Shape2D(x)
		init := rand.NormFloat64()*0.5 + 0.5 - drop.Ratio
		randomArray := gmat.MakeInit(n, m, init)
		drop.Mask = gmat.Apply(randomArray, layer.MaskFunc)
		return gmat.Mul(x, drop.Mask)
	} else {
		return gmat.MulE(x, 1.0-drop.Ratio)
	}
}

func (drop *Dropout) Backward(dout gmat.Tensor) gmat.Tensor {
	return gmat.Mul(dout, drop.Mask)
}

func Accuracy(x gmat.Tensor, t gmat.Tensor) float64 {
	indexArray := gmat.ArgMaxCol(x)
	n, _ := gmat.Shape2D(x)
	//nt, mt := gmat.Shape2D(t)
	accuracy := 0
	for i := 0; i < n; i++ {
		if t.CPU[i][indexArray[i][0]] > 0.9 {
			accuracy += 1
		}
	}
	return float64(accuracy) / float64(n)
}

type LayerInterface interface {
	Forward(gmat.Tensor) gmat.Tensor
	Backward(gmat.Tensor) gmat.Tensor
	sgdUpdate(*SGD)
	momentumUpdate(*Momentum)
}

func Layerinit() interface{} {
	return []LayerInterface{}
}

func ForwardLayer(layer []LayerInterface, x gmat.Tensor) gmat.Tensor {
	for _, v := range layer {
		x = v.Forward(x)
	}
	return x
}

func BackLayer(layer []LayerInterface, dout gmat.Tensor) gmat.Tensor {
	for i := len(layer) - 1; i >= 0; i-- {
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

func DensePrint(x gmat.Tensor, name string) {
	common.DensePrint(x, name)
}

func AvePrint(x gmat.Tensor, name string) {
	common.AvePrint(x, name)
}

// update

func SGDUpdateLayer(layer []LayerInterface, sgd *SGD) {
	for _, v := range layer {
		v.sgdUpdate(sgd)
	}
	return
}

func MomentumUpdateLayer(layer []LayerInterface, m *Momentum) {
	for _, v := range layer {
		v.momentumUpdate(m)
	}
	return
}

func (dense *Dense) sgdUpdate(sgd *SGD) {
	dense.W = gmat.Sub(dense.W, gmat.MulE(dense.Dw, sgd.LearningRate))
	dense.B = gmat.Sub(dense.B, gmat.MulE(dense.Db, sgd.LearningRate))
	return
}

func (relu *Relu) sgdUpdate(sgd *SGD) {
	return
}

func (drop *Dropout) sgdUpdate(sgd *SGD) {
	return
}

func (dense *Dense) momentumUpdate(m *Momentum) {
	dense.W = momentumCalc(m, dense.W, dense.Dw, &dense.Vw)
	dense.B = momentumCalc(m, dense.B, dense.Db, &dense.Vb)
	common.DenseCheck(dense.W, "momentum dense.W")
	common.DenseCheck(dense.B, "momentum dense.B")
}

func (relu *Relu) momentumUpdate(m *Momentum) {
	return
}

func (drop *Dropout) momentumUpdate(m *Momentum) {
	return
}

func momentumCalc(m *Momentum, layerParam gmat.Tensor, layerDelta gmat.Tensor, v *gmat.Tensor) gmat.Tensor {
	if (*v).CPU == nil {
		r, c := gmat.Shape2D(layerParam)
		*v = gmat.Make2D(r, c)
	}
	av := gmat.MulE(*v, m.MomentumRate)
	dl := gmat.MulE(layerDelta, m.LearningRate)
	*v = gmat.Sub(av, dl)
	layerParam = gmat.Add(layerParam, *v)
	return layerParam
}

package gdeep

import (
	"github.com/kuroko1t/GoMNIST"
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
	"github.com/kuroko1t/gdeep/layer"
)

type Relu layer.Relu
type Sigmoid layer.Sigmoid
type Dense layer.Dense
type SoftmaxWithLoss layer.SoftmaxWithLoss
type SGD layer.SGD
type Momentum layer.Momentum

func (relu *Relu) Forward(x [][]float64) [][]float64 {
	relu.Mask = gmat.Apply(x, layer.MaskFunc)
	x = gmat.Mul(x, relu.Mask)
	common.DenseCheck(x, "Relu forward output")
	return x
}

func (relu *Relu) Backward(dout [][]float64) [][]float64 {
	dout = gmat.Mul(dout, relu.Mask)
	common.DenseCheck(dout, "Relu backward output")
	return dout
}

func (sigmoid *Sigmoid) Forward(x [][]float64) [][]float64 {
	x = gmat.Apply(x, layer.Sigmoid_f)
	sigmoid.Out = x
	return x
}

func (sigmoid *Sigmoid) Backward(dout [][]float64) [][]float64 {
	dx := gmat.Apply(dout, layer.Sigmoid_b)
	dx = gmat.Mul(dx, sigmoid.Out)
	dx = gmat.Mul(dx, dout)
	return dx
}

func (dense *Dense) Forward(x [][]float64) [][]float64 {
	common.DenseCheck(x, "dense forward input")
	dense.X = x
	c := gmat.Dot(x, dense.W)
	c = gmat.Add(c, dense.B)
	common.DenseCheck(c, "dense forward output")
	return c
}

func (dense *Dense) Backward(dout [][]float64) [][]float64 {
	wt := gmat.T(dense.W)
	dx := gmat.Dot(dout, wt)
	xt := gmat.T(dense.X)
	dense.Dw = gmat.Dot(xt, dout)
	dense.Db = gmat.SumRow(dout)
	common.DenseCheck(dx,"dense backward output")
	return dx
}

func (softmaxWithLoss *SoftmaxWithLoss) Forward(x [][]float64, t [][]float64) [][]float64 {
	softmaxWithLoss.T = t
	softmaxWithLoss.Y = layer.Softmax(x)
	softmaxWithLoss.Loss = layer.CrossEnrtopyError(softmaxWithLoss.Y, softmaxWithLoss.T)
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

func Accuracy(x [][]float64, t [][]float64) float64{
	indexArray := gmat.ArgMaxCol(x)
	n, _ := gmat.Shape2D(x)
	//nt, mt := gmat.Shape2D(t)
	accuracy := 0
	for i := 0 ; i < n ; i++ {
		if t[i][indexArray[i][0]] > 0.9 {
			accuracy += 1
		}
	}
	return float64(accuracy) / float64(n)
}

type LayerInterface interface {
	Forward([][]float64) [][]float64
	Backward([][]float64) [][]float64
	sgdUpdate(*SGD)
	momentumUpdate(*Momentum)
}

func Layerinit() interface{} {
	return []LayerInterface{}
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

func DensePrint(x [][]float64, name string) {
	common.DensePrint(x, name)
}

func AvePrint(x [][]float64, name string) {
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

func (dense *Dense)momentumUpdate(m *Momentum) {
	dense.W = momentumCalc(m, dense.W, dense.Dw, &dense.Vw)
	dense.B = momentumCalc(m, dense.B, dense.Db, &dense.Vb)
	common.DenseCheck(dense.W, "momentum dense.W")
	common.DenseCheck(dense.B, "momentum dense.B")
}

func (relu *Relu) momentumUpdate(m *Momentum) {
	return
}

func momentumCalc(m *Momentum, layerParam [][]float64, layerDelta [][]float64, v *[][]float64) [][]float64 {
	if *v == nil {
		*v = gmat.Make(len(layerParam),len(layerParam[0]))
	}
	av := gmat.MulE(*v , m.MomentumRate)
	dl := gmat.MulE(layerDelta, m.LearningRate)
	*v = gmat.Sub(av ,dl)
	layerParam = gmat.Add(layerParam, *v)
	return layerParam
}

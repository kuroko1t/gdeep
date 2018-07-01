package main

import (
	"fmt"
	"math"
	"math/rand"
    "github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
//	"encoding/binary"
//	"reflect"
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
	//fmt.Println(x)
	//fmt.Println(relu.mask)
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
	//fmt.Println("koko", sigmoid.out)
	dx.Apply(sigmoid_b, dout)
	//fmt.Println("koko", sigmoid.out)
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
	//fmt.Println(c)
	return &c
}

func (affine *Affine) backward(dout *mat.Dense) *mat.Dense {
	wt := affine.w.T()
	var dx mat.Dense
	dx.Mul(dout, wt)
	xt := affine.x.T()
	affine.dw.Mul(xt, dout)
	affine.db = sumRow(dout)
	return &dx
}

func sumRow(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	sumArray := make([]float64, r*c)
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

func (softmaxWithLoss *SoftmaxWithLoss) backward(dout *mat.Dense) (*mat.Dense) {
	batchSize, c := softmaxWithLoss.t.Dims()
	batchSizeArray := make([]float64, batchSize* c)
	for i, _ := range batchSizeArray {
		batchSizeArray[i] = float64(batchSize)
	}
	batchSizeMatrix := mat.NewDense(batchSize, c, batchSizeArray)
	submat := mat.NewDense(batchSize, c, nil)
	submat.Sub(softmaxWithLoss.y, softmaxWithLoss.t)
	submat.DivElem(submat, batchSizeMatrix)
	return submat
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
	forward(*mat.Dense) *mat.Dense
	backward(*mat.Dense) *mat.Dense
}

func Update(layer []ForwardInterface, x *mat.Dense) *mat.Dense {
	for _, v := range layer {
		x = v.forward(x)
	}
	return x
}

func BackLayer(layer []ForwardInterface, dout *mat.Dense) *mat.Dense {
	for i := len(layer)-1; i >=0; i-- {
		f := layer[i]
		dout = f.backward(dout)
	}
	return dout
}

func constMulElem(x *mat.Dense, constNum float64) (y *mat.Dense) {
	r, c := x.Dims()
	xData := make([]float64, r*c)
	for i, _ := range  xData{
		xData[i] = constNum
	}
	y = mat.NewDense(r, c, xData)
	return y
}

func OneHot(x int, size int) (y []float64) {
	y = make([]float64, size)
	if x != 0 {
		y[x-1] = 1
	}
	return y
}

func MnistBatch(sweep *GoMNIST.Sweeper, batchSize int) ([]float64 ,[]float64) {
	outputSize := 10
	image := make([]float64, 784*batchSize)
	//label := make([]float64, 10*batchSize)
	var label []float64
	for i:= 0; i < batchSize ; i++ {
		if i == 0 {
			image_tmp, label_tmp, _ := sweep.Next()
			for o := 0 ; o < len(image_tmp) ; o ++ {
				image[o] = float64(image_tmp[o])
			}
			labelOneHotTmp := OneHot(int(label_tmp),outputSize)
			fmt.Println(labelOneHotTmp)
			//label = append(label,labelOneHotTmp)
			//for i := 0 ; i < len(labelOneHottmp) ; i ++ {
			// 	label[i] = float64(el_tmp[i])
			//}
		} else {
			image_tmp, label_tmp, _ := sweep.Next()
			//fmt.Println(label_tmp)
			for o := 0 ; i < len(image_tmp) ; o ++ {
				image[o+i] = float64(image_tmp[o+i])
			}
			//labelOneHotTmp := OneHot(int(label_tmp),outputSize)
			//label = append(label,labelOneHotTmp)
			//for i := 0 ; i < len(label_tmp) ; i ++ {
			// 	label_tmpFloat[i] = float64(label_tmp[i])
			// 	label = append(label,label_tmpFloat)
			//}
		}
	}
	return image, label
}

func main() {
	//zero.Apply(add1, zero)
	batchSize := 3
	inputSize := 784
	hiddenSize := 2
	outputSize := 10
	learningRate := 0.01

	w0Data := make([]float64, inputSize*hiddenSize)
	b0Data := make([]float64, batchSize*hiddenSize)
	w1Data := make([]float64, hiddenSize*outputSize)
	b1Data := make([]float64, batchSize*outputSize)
	w0 := mat.NewDense(inputSize, hiddenSize, randomArray(w0Data))
	b0 := mat.NewDense(batchSize, hiddenSize, randomArray(b0Data))
	w1 := mat.NewDense(hiddenSize, outputSize, randomArray(w1Data))
	b1 := mat.NewDense(batchSize, outputSize, randomArray(b1Data))

	layer := []ForwardInterface{}
	affine1 := &Affine{w0, b0, w0, w0, b0}
	relu1 := &Relu{b0}
	affine2 := &Affine{w1, b1, w1, w1, b1}
	layer = append(layer, affine1)
	layer = append(layer, relu1)
	layer = append(layer, affine2)

	train, _, _ := GoMNIST.Load("./data")
	sweeper := train.Sweep()
	xBatch , label:= MnistBatch(sweeper, 2)

	//xBAtch := mat.NewDense(2, inputSize, float64(xBatch))
	fmt.Println(xBatch)
	fmt.Println(label)
	//x_batch := append(x_batch,)
	//for {
	// 	image, label , present := sweeper.Next()
	// 	fmt.Println("label",label,OneHot(int(label), 10))
	// 	if !present {
	// 		break
	// 	}
	//}
	for i :=0 ; i < 2 ; i ++ {
		xData := make([]float64, batchSize*inputSize)
		tData := make([]float64, batchSize*outputSize)
		x := mat.NewDense(batchSize, inputSize, randomArray(xData))
		t := mat.NewDense(batchSize, outputSize, randomArray(tData))
		x = Update(layer, x)
		softmaxWithLoss := SoftmaxWithLoss{}
		loss := softmaxWithLoss.forward(x, t)
		fmt.Println(loss)
		dout := mat.NewDense(batchSize, outputSize, randomArray(tData))
		dout =  softmaxWithLoss.backward(dout)
		dout = BackLayer(layer,dout)
		affine1.w.Sub(affine1.w, constMulElem(affine1.dw, learningRate))
		affine1.b.Sub(affine1.b, constMulElem(affine1.db ,learningRate))
		affine2.w.Sub(affine2.w, constMulElem(affine2.dw ,learningRate))
		affine2.b.Sub(affine2.b, constMulElem(affine2.db ,learningRate))
	}
}

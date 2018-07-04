package main

import (
	"fmt"
	"math"
	"math/rand"
    "github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
	"github.com/kuroko1t/gdeep/common"
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
	//fmt.Println("relu forward", x)
	common.DenseCheck(x, "Relu forward output")
	return x
}

func (relu *Relu) backward(dout *mat.Dense) *mat.Dense {
	//fmt.Println("relu.mask", relu.mask)
	dout.MulElem(dout, relu.mask)
	//fmt.Println("relu backward", dout)
	common.DenseCheck(dout, "Relu backward output")
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
	//fmt.Println("sigmoid forward", x)
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
	//fmt.Println("sigmoid backward", dx)
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
	//fmt.Println("affine input",x)
	//fmt.Println("affine forward affine.w",affine.w)
	common.DenseCheck(x, "affine forward input")
	affine.x = x
	var c mat.Dense
	c.Mul(x, affine.w)
	//fmt.Println("affine forward x", x)
	//fmt.Println("affine in forward affine.w", affine.w)
	//fmt.Println("affine forward c", c)
	//fmt.Println("affine forward affine.b", affine.b)
	c.Add(&c, affine.b)
	//fmt.Println("affine forward",c)
	common.DenseCheck(&c, "affine forward output")
	return &c
}

func (affine *Affine) backward(dout *mat.Dense) *mat.Dense {
	//fmt.Println("affine.w", affine.w)
	wt := affine.w.T()
	var dx mat.Dense
	dx.Mul(dout, wt)
	//fmt.Println("affine back dx", dx)
	xt := affine.x.T()
	affine.dw.Mul(xt, dout)
	//fmt.Println("affine backward affine.dw", affine.dw)
	affine.db = sumRow(dout)
	//common.DenseCheck(affine.db,"affine.db")
	common.DenseCheck(&dx,"affine backward output")
	//fmt.Println("affine backward", dx)
	return &dx
}

func sumRow(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	sumArray := make([]float64, r*c)
	for j := 0; j < c; j++ {
		sumValue := 0.0
		for i := 0; i < r; i++ {
			sumValue += m.At(i, j)
		}
		for i := 0; i < r; i++ {
			sumArray[i*c + j] = sumValue
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
		}
		for i := 0; i < c; i++ {
			sumArray[i+j*c] = sumValue
		}
	}
	sums := mat.NewDense(r, c, sumArray)
	return sums
}

// softmax_withloss

func (softmaxWithLoss *SoftmaxWithLoss) forward(x *mat.Dense, t *mat.Dense) *mat.Dense {
	softmaxWithLoss.t = t
	//fmt.Println("softmax before",x)
	softmaxWithLoss.y = softmax(x)
	//fmt.Println("softmax after",softmaxWithLoss.y)
	softmaxWithLoss.loss = crossEnrtopyError(softmaxWithLoss.y, softmaxWithLoss.t)
	//fmt.Println("softmaxwithloss forward", softmaxWithLoss.loss)
	common.DenseCheck(softmaxWithLoss.loss, "softmaxwithloss forward");
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
	//fmt.Println("softmaxwithloss backward", submat)
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
	//fmt.Println("crossEnrtopy b",y)
	y.Apply(crossEnrtopy, y)
	y.MulElem(t, y)
	common.DenseCheck(y, "crossEnrtopyError before sumcol");
	//fmt.Println(y)
	y.Apply(minus, sumCol(y))
	//fmt.Println(y)
	//fmt.Println("crossEnrtopy a",y)
	common.DenseCheck(y, "crossEnrtopyError outout");
	return y
}

func crossEnrtopy(i, j int, v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

func randomArray(data []float64) []float64 {
	for i := range data {
		data[i] = rand.NormFloat64() * 0.01
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

func constMulElem(x *mat.Dense, constNum float64) (*mat.Dense) {
	r, c := x.Dims()
	xData := make([]float64, r*c)
	for i, _ := range  xData{
		xData[i] = constNum
	}
	y := mat.NewDense(r, c, xData)
	//common.DenseCheck(y,"const learningrate")
	x.MulElem(x, y)
	//common.DenseCheck(x,"const output")
	return x
}

func constMul(constnum float64) {

}

func OneHot(x int, size int) (y []float64) {
	y = make([]float64, size)
	if x != 0 {
		y[x-1] = 1
	}
	return y
}

func MnistBatch(sweep **GoMNIST.Sweeper, batchSize int) ([]float64 ,[]float64, bool) {
	outputSize := 10
	imagenum := 784
	image := make([]float64, imagenum*batchSize)
	var label []float64
	present := true
	for i:= 0; i < batchSize ; i++ {
		if i == 0 {
			image_tmp, label_tmp, present := (*sweep).Next()
			if !present {
				return make([]float64,2), make([]float64,2), present
			}
			for o := 0 ; o < len(image_tmp) ; o++ {
				image[o] = float64(image_tmp[o])/255
			}
			label = OneHot(int(label_tmp),outputSize)
		} else {
			image_tmp, label_tmp, present := (*sweep).Next()
			if !present {
				return make([]float64,2), make([]float64,2), present
			}
			for o := 0 ; o < len(image_tmp) ; o++ {
				image[o+i*imagenum] = float64(image_tmp[o])/255
			}
			labelOneHotTmp := OneHot(int(label_tmp),outputSize)
			label = append(label,labelOneHotTmp...)
		}
	}
	return image, label ,present
}

func main() {
	batchSize := 100
	inputSize := 784
	hiddenSize := 50
	outputSize := 10
	learningRate := 0.01

	w0Data := make([]float64, inputSize*hiddenSize)
	//b0Data := make([]float64, batchSize*hiddenSize)
	w1Data := make([]float64, hiddenSize*outputSize)
	//b1Data := make([]float64, batchSize*outputSize)
	w0 := mat.NewDense(inputSize, hiddenSize, randomArray(w0Data))
	b0 := mat.NewDense(batchSize, hiddenSize, nil)
	w1 := mat.NewDense(hiddenSize, outputSize, randomArray(w1Data))
	b1 := mat.NewDense(batchSize, outputSize, nil)
	//b1 := mat.NewDense(batchSize, outputSize, randomArray(b1Data))
	//fmt.Println("w0",w0)
	//fmt.Println("b0",b0)
	//fmt.Println("w1",w1)
	//fmt.Println("b1",b1)
	//tmpdata := []float64{2,3,4,5,5,5}//make([]float64, 2*3)
	//tmp := mat.NewDense(2, 3, tmpdata)
	//fmt.Println("tmp",tmp)
	//fmt.Println("sumCol",sumCol(tmp))
	common.DenseCheck(w0,"w0")
	common.DenseCheck(b0,"b0")
	common.DenseCheck(w1,"w1")
	common.DenseCheck(b1,"b1")
	layer := []ForwardInterface{}
	affine1 := &Affine{w0, b0, w0, w0, b0}
	relu1 := &Relu{b0}
	affine2 := &Affine{w1, b1, w1, w1, b1}
	layer = append(layer, affine1)
	layer = append(layer, relu1)
	layer = append(layer, affine2)

	train, _, _ := GoMNIST.Load("./data")
	sweeper := train.Sweep()
	//fmt.Printf("out sweep:%p",sweeper)
	//for {
	// 	_, label, present := sweeper.Next()
	// 	fmt.Println(label)
	// 	if !present {
	// 		break
	// 	}
	//}
	for i :=0 ; i < 1 ; i ++ {
		fmt.Println("[iteration:",i,"]")
		//xData := make([]float64, batchSize*inputSize)
		//tData := make([]float64, batchSize*outputSize)
		xData, tData, present:= MnistBatch(&sweeper, batchSize)
		if !present {
			sweeper = train.Sweep()
			xData, tData, present= MnistBatch(&sweeper, batchSize)
		}
		//fmt.Println("tData",tData)
		//fmt.Println("input data",xData)
		x := mat.NewDense(batchSize, inputSize, xData)
		t := mat.NewDense(batchSize, outputSize, tData)
		common.DenseCheck(x, "x");
		//x := mat.NewDense(batchSize, inputSize, randomArray(xData))
		//t := mat.NewDense(batchSize, outputSize, randomArray(tData))
		x = Update(layer, x)
		softmaxWithLoss := SoftmaxWithLoss{}
		loss := softmaxWithLoss.forward(x, t)
		common.DensePrint(loss, "loss");
		//fmt.Println("loss",loss)
		//fmt.Println("sum:",sumCol(loss))
		dout := mat.NewDense(batchSize, outputSize, randomArray(tData))
		dout =	softmaxWithLoss.backward(dout)
		dout = BackLayer(layer,dout)
		//fmt.Println("affine1.w",affine1.w)
		//fmt.Println("constMulElem",constMulElem(affine1.dw, learningRate))
		//fmt.Println("affine1.dw",affine1.dw)
		common.DenseCheck(affine1.w,"affine1.w")
		common.DenseCheck(affine1.b,"affine1.b")
		common.DenseCheck(affine2.w,"affine2.w")
		common.DenseCheck(affine2.b,"affine2.b")
		common.DenseCheck(affine1.dw,"affine1.dw")
		common.DenseCheck(affine1.db,"affine1.db")
		common.DenseCheck(affine2.dw,"affine2.dw")
		common.DenseCheck(affine2.db,"affine2.db")
		//common.DenseCheck(constMulElem(affine2.db ,learningRate), "constMulElem affine2.db")
		affine1.w.Sub(affine1.w, constMulElem(affine1.dw, learningRate))
		affine1.b.Sub(affine1.b, constMulElem(affine1.db ,learningRate))
		affine2.w.Sub(affine2.w, constMulElem(affine2.dw ,learningRate))
		affine2.b.Sub(affine2.b, constMulElem(affine2.db ,learningRate))
		common.DenseCheck(affine1.w,"affine1.w")
		common.DenseCheck(affine1.b,"affine1.b")
		common.DenseCheck(affine2.w,"affine2.w")
		common.DenseCheck(affine2.b,"affine2.b")
		//fmt.Println("affine1.b",affine1.b)
		//fmt.Println("affine2.w",affine2.w)
		//fmt.Println("affine2.b",affine2.b)
	}
}

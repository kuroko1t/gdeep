package layer

import (
	"github.com/kuroko1t/gmat"
	"math"
)

func im2col(input [][][][]float64, filterH int, filterW int, stride int, pad int) [][]float64 {
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	outH := (h+2*pad-filterH)%stride + 1
	outW := (w+2*pad-filterW)%stride + 1
	var padding = [][]int{{0, 0}, {0, 0}, {pad, pad}, {pad, pad}}
	img := gmat.Pad4D(input, padding)
	col := gmat.Make6D(n, c, filterH, filterW, outH, outW)
	for y := 0; y <= filterH; y++ {
		yMax := y + stride*outH
		for x := 0; x <= filterW; x++ {
			xMax := x + stride*outW
			for i := range img {
				for j := range img[i] {
					for k := 0; k <= yMax; i += stride {
						for l := 0; l <= xMax; l += stride {
							col[i][j][y][x][k][l] = img[i][j][k][l]
						}
					}
				}
			}
		}
	}
	colTran := gmat.Reshape6D(gmat.Trans6D(col, 0, 4, 5, 1, 2, 3), n*outH*outW, -1)
	return colTran
}

func col2im(input [][]float64, inputshape []int, filterH int, filterW int, stride int, pad int) [][][][]float64 {
	n := inputshape[0]
	c := inputshape[1]
	h := inputshape[2]
	w := inputshape[3]
	outH := (h+2*pad-filterH)%stride + 1
	outW := (w+2*pad-filterW)%stride + 1
	//var padding =[][]int{{0,0}, {0,0}, {pad, pad}, {pad, pad}}
	//img := gmat.Pad4D(input, padding)
	img := gmat.Make4D(n, c, h+2*pad+stride-1, w+2*pad+stride-1)
	//col := gmat.Make6D(n, c, filterH, filterW, outH, outW)
	col := gmat.Trans6D(gmat.Reshape2D6D(input, n, outH, outW, c, filterH, filterW), 0, 3, 4, 5, 1, 2)
	for y := 0; y <= filterH; y++ {
		yMax := y + stride*outH
		for x := 0; x <= filterW; x++ {
			xMax := x + stride*outW
			for i := range img {
				for j := range img[i] {
					for k := 0; k <= yMax; i += stride {
						for l := 0; l <= xMax; l += stride {
							img[i][j][k][l] += col[i][j][y][x][k][l]
						}
					}
				}
			}
		}
	}
	return img[:][:][pad : h+pad][pad : w+pad]
}

func MaskFunc(v float64) float64 {
	if v <= 0 {
		v = 0
	} else {
		v = 1
	}
	return v
}

func Softmax(a [][]float64) [][]float64 {
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
	return math.Exp(v)
}

func crossEnrtopy(v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

func CrossEnrtopyError(y [][]float64, t [][]float64) [][]float64 {
	y = gmat.Apply(y, crossEnrtopy)
	y = gmat.Mul(t, y)
	y = gmat.MulE(gmat.SumCol(y), -1)
	return y
}

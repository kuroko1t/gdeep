package gdeep

import (
	//"math"
	//"github.com/petar/GoMNIST"
	//"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
)

type Conv struct {
	W  [][][][]float64
	B  [][]float64
	stride int
	pad int
	x [][][][]float64
	col [][]float64
	colW [][]float64
	dw [][]float64
	db [][]float64
}

func (conv *Conv) Forward(x [][][][]float64) [][][][]float64 {
	fn, _, fh, fw := gmat.Shape4D(conv.W)
	n, _, h, w := gmat.Shape4D(x)
	outH := 1 + (h + 2 * conv.pad - fh) / conv.stride
	outW := 1 + (w + 2 * conv.pad - fw) / conv.stride

	col := im2col(x, fh, fw, conv.stride, conv.pad)
	colW := gmat.T(gmat.Reshape4D(conv.W, fn, -1))

	out := gmat.Add(gmat.Dot(col, colW), conv.B)
	out4d := gmat.Trans4D(gmat.Reshape2D(out, n, outH, outW, -1), 0, 3, 1, 2)

	conv.x = x
	conv.col = col
	conv.colW = colW
	return out4d
}

func (conv *Conv) Backward(dout [][][][]float64) [][][][]float64 {
	fn, fc, fh, fw := gmat.Shape4D(conv.W)
	dout = gmat.Reshape4D(gmat.Trans4D(dout, 0, 2, 3, 1),-1 , fn)

	conv.db = gmat.SumRow(dout)

	return dout
}

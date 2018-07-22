// +build debug

package layer

import (
	"github.com/kuroko1t/gmat"
)

type Data data.Data

type Conv struct {
	W  gmat.Data
	B  [][]float32
	stride int
	pad int
	x [][][][]float32
	col [][]float32
	colW [][]float32
	dw [][][][]float32
	db [][]float32
}

func (conv *Conv) Forward(x [][][][]float32) [][][][]float32 {
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

func (conv *Conv) Backward(dout [][][][]float32) [][][][]float32 {
	fn, fc, fh, fw := gmat.Shape4D(conv.W)
	dout2D := gmat.Reshape4D(gmat.Trans4D(dout, 0, 2, 3, 1),-1 , fn)
	conv.db = gmat.SumRow(dout2D)
	dw2D := gmat.Dot(gmat.T(conv.col),dout2D)
	conv.dw = gmat.Reshape2D(gmat.Trans2D(dw2D, 1, 0), fn, fc, fh, fw)

	dcol := gmat.Dot(dout2D, gmat.T(conv.colW))
	n,c,h,w := gmat.Shape4D(conv.x)
	xshape := []int{n,c,h,w}
	dx := col2im(dcol, xshape, fh, fw, conv.stride, conv.pad)
	return dx
}


type Pooling struct {
	poolH int
	poolW int
	stride int
	pad int
	x [][][][]float32
	argMax [][]int
	col [][]float32
}

func (pool *Pooling) Forward(x [][][][]float32) [][][][]float32 {
	n,c,h,w := gmat.Shape4D(x)
	outH := 1 + (h - pool.poolH) / pool.stride
	outW := 1 + (w - pool.poolW) / pool.stride

	col2D := im2col(x, pool.poolH, pool.poolW, pool.stride, pool.pad)
	col := gmat.Reshape2D2D(col2D,-1, pool.poolH* pool.poolW)
	out2D := gmat.MaxCol(col)
	out := gmat.Trans4D(gmat.Reshape2D(out2D, n, outH, outW, c), 0, 3, 1, 2)
	pool.x = x
	pool.argMax = gmat.ArgMaxCol(col)
	return out
}

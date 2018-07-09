package gdeep

import (
	"math"
	"github.com/petar/GoMNIST"
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
)

func im2col(input [][][][]float64, filterH int, filterW int, stride int, pad int) {
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	outH := (h + 2*pad - filterH) % (stride + 1)
	outW = (w + 2*pad - filterW) % (stride + 1)
	var padding =[][]int{{0,0}, {0,0}, {pad, pad}, {pad, pad}}
	img := gmat.Pad4D(input_data, padding)
	col := gmat.Make4D((N, C, filterH, filterW, outH, outW))
	for y:=0 ;y <= filterH; y++ {
		yMax := y + stride*outH
		for x:=0 ;x <= filterW; x++ {
			xMax = x + stride*outW
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	}
}

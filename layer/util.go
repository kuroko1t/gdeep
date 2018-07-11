package gdeep

import (
	"github.com/kuroko1t/gmat"
)

func im2col(input [][][][]float64, filterH int, filterW int, stride int, pad int) [][]float64{
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	outH := (h + 2*pad - filterH) % (stride + 1)
	outW := (w + 2*pad - filterW) % (stride + 1)
	var padding =[][]int{{0,0}, {0,0}, {pad, pad}, {pad, pad}}
	img := gmat.Pad4D(input, padding)
	col := gmat.Make6D(n, c, filterH, filterW, outH, outW)
	for y:=0 ;y <= filterH; y++ {
		yMax := y + stride*outH
		for x:=0 ;x <= filterW; x++ {
			xMax := x + stride*outW
			for i := range img {
				for j := range img[i] {
					for k :=0; k <=yMax; i += stride {
						for l :=0; l <=xMax; l += stride {
							col[i][j][y][x][k][l]= img[i][j][k][l]
						}
					}
				}
			}
		}
	}
	colTran := gmat.Reshape6D(gmat.Trans6D(col, 0,4,5,1,2,3),n*outH*outW,-1)
	return colTran
}

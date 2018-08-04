// +build debug
// Copyright 2018 kurosawa. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

package layer

func im2col(input [][][][]float32, filterH int, filterW int, stride int, pad int) [][]float32 {
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

func col2im(input [][]float32, inputshape []int, filterH int, filterW int, stride int, pad int) [][][][]float32 {
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

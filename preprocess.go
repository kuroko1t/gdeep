package gdeep

import (
	"github.com/kuroko1t/gdeep/common"
	"github.com/kuroko1t/gmat"
)

func ImageRead2GrayNorm(imgpath string) (t gmat.Tensor) {
	floatArray := common.ImageRead2GrayNorm(imgpath)
	z := [][]float64{}
	z = append(z, floatArray)
	t.CPU = z
	return t
}

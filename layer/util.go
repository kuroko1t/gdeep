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

import (
	"github.com/kuroko1t/gmat"
	"math"
)

func MaskFunc(v float64) float64 {
	if v <= 0 {
		v = 0
	} else {
		v = 1
	}
	return v
}

func Softmax(a gmat.Tensor) gmat.Tensor {
	a = gmat.Apply(a, Exp)
	n := len(a.CPU[0])
	sumExp := gmat.SumCol(a)
	sumExpCast := gmat.Cast(sumExp, n)
	sumExp = gmat.Div(a, sumExpCast)
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
	return float64(math.Exp(float64(v)))
}

func crossEnrtopy(v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

func CrossEnrtopyError(y, t gmat.Tensor) gmat.Tensor {
	y = gmat.Apply(y, crossEnrtopy)
	y = gmat.Mul(t, y)
	crossError := gmat.MulE(gmat.SumCol(y), -1)
	return crossError
}

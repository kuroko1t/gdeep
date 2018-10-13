// +build gpu

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

func Softmax(a gmat.Tensor) gmat.Tensor {
	a = gmat.Exp(a, 0.0, 0.0)
	n := len(a.CPU[0])
	sumExp := gmat.SumCol(a)
	sumExpCast := gmat.Cast(sumExp, n)
	sumExp = gmat.Div(a, sumExpCast)
	return sumExp
}

func crossEnrtopy(v float64) float64 {
	delta := 0.000001
	return math.Log(v + delta)
}

func CrossEnrtopyError(y, t gmat.Tensor) gmat.Tensor {
	delta := 0.000001
	y = gmat.Log(y, delta)
	y = gmat.Mul(t, y)
	crossError := gmat.MulE(gmat.SumCol(y), -1)
	return crossError
}

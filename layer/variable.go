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
)

type Sigmoid struct {
	Out gmat.Tensor
}

type Dense struct {
	W  gmat.Tensor
	B  gmat.Tensor
	X  gmat.Tensor
	Dw gmat.Tensor
	Db gmat.Tensor
	Vw gmat.Tensor
	Vb gmat.Tensor
}

type Dropout struct {
	Mask  gmat.Tensor
	Train bool
	Ratio float64
}

type SoftmaxWithLoss struct {
	Loss gmat.Tensor
	Y    gmat.Tensor
	T    gmat.Tensor
}

type Relu struct {
	Mask gmat.Tensor
}

type SGD struct {
	LearningRate float64
}

type Momentum struct {
	LearningRate float64
	MomentumRate float64
}

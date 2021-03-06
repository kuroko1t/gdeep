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
package gdeep

import (
	"encoding/gob"
	"log"
	"os"
	"reflect"
)

func Saver(p []LayerInterface, saveName string) {
	f, err := os.Create(saveName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, v := range p {
		gob.Register(v)
	}
	if err := enc.Encode(&p); err != nil {
		log.Fatal(err)
	}
}

func Restore(fileName string, p *[]LayerInterface) {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	gob.Register(&Dense{})
	gob.Register(&Relu{})
	gob.Register(&Dropout{})
	gob.Register(&SoftmaxWithLoss{})
	if err := dec.Decode(p); err != nil {
		log.Fatal("decode error:", err)
	}
	for i := 0; i < len(*p); i++ {
		if reflect.TypeOf(&Dropout{}) == reflect.TypeOf((*p)[i]) {
			dropout := (*p)[i].(*Dropout)
			dropout.Train = false
		}
	}
}

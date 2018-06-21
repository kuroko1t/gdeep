package main

import "fmt"
import "gonum.org/v1/gonum/mat"

type Affine struct {
	dw float64
	db float64
	w float64
	b float64
}

func forward(x *mat.Dense) (*mat.Dense) {
	//x.Apply(minus,x)
	//var m mat.Dense
	fmt.Print(x)
	x.Exp(x)
	return x
}

func add1(i, j int, v float64) float64 {
	return v + 1
}

func minus(i, j int, v float64) float64 {
	return v * -1
}

func main() {
	zero := mat.NewDense(3, 5, nil)
	zero.Apply(add1,zero)
	forward(zero)
	fmt.Print(zero)
}

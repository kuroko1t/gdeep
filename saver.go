package gdeep

import (
	"log"
	"os"
	"encoding/gob"
	//"fmt"
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

func Restore(fileName string, p interface{}){
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	//var q
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&p); err != nil {
		log.Fatal("decode error:", err)
	}
}

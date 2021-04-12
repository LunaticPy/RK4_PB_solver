package main

import (
	"fmt"

	"./num_eu"
)

//

func main() {
	num_eu.Init_in_value(0.5, 0, 5)

	data, _ := num_eu.Read_MD_data()
	fmt.Println(data[1])
}

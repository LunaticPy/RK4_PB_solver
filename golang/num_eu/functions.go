package num_eu

import (
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
)

//типы функций
//-----------------------------------------

type Anfunc = func(r float64) float64
type Diffunc = func(r, y, dy float64) float64

//-----------------------------------------

func Read_MD_data() (data [][][2]float64, fname []string) {
	/*
	   Чтение данных молекулярной динамики
	   mol_d - путь к МД результатам
	   data - [list] результат
	*/
	var (
		mol_d   string = "C:\\Users\\1\\Downloads\\densprof_coul_5eV\\densprof_coul_5eV\\"
		loc_val [2]float64
	)

	f_list, _ := ioutil.ReadDir(mol_d)
	for _, i := range f_list {
		if i.Name()[len(i.Name())-3:] == "dat" { // go reading the file
			loc_dat := make([][2]float64, 0)
			file, err := os.Open(mol_d + i.Name())
			if err != nil {
				fmt.Println(err)
				os.Exit(1)
			}
			for {
				for j := 0; j < 2; j++ {
					_, err = fmt.Fscanf(file, "%f", &loc_val[j])
					if err == io.EOF {
						break // stop reading the file
					}
				}
				if err == io.EOF {
					break // stop reading the file
				}
				fname = append(fname, i.Name())
				loc_dat = append(loc_dat, loc_val)
			}
			data = append(data, loc_dat)
		}
	}

	return
}

//==============================================================================================
//   функции на граничные условия
func phi_0_DH(rho, R float64) float64 {
	/*
		Внутренняя область аналитического решения
		-Обезразмерянное
	*/
	return -rho*(1+R)*math.Exp(-R) + rho
}
func phi_s_DH(rho, R float64) float64 {
	/*
		Внешняя область аналитического решения
		-Обезразмерянное
	*/
	return -rho / R * math.Exp(-R) * (math.Sinh(R) - R*math.Cosh(R))
}

//   расширение функций на граничные условия

func pot_DH(r []float64, rho, R float64) []float64 {
	/*
	    Полное аналитическое решение
	   -Обезразмерянное
	*/
	var rez []float64
	var rezf Anfunc = func(r float64) float64 {

		if r < R {
			return math.Sinh(r)/r*(phi_0_DH(rho, R)-rho) + rho
		} else {
			return R * phi_s_DH(rho, R) * math.Exp(R-r) / r
		}
	}
	for _, r_i := range r {
		rez = append(rez, rezf(r_i))
	}
	return rez
}
func dpot_DH(r []float64, rho, R float64) []float64 {
	/*
	   Производная полного аналитического решения
	   -Обезразмерянное
	*/
	var rez []float64
	var rezf Anfunc = func(r float64) float64 {
		if r < R {
			return -rho * (1 + R) * math.Exp(-R) * (r*math.Cosh(r) - math.Sinh(r)) / r / r
		} else {
			return -rho * (1 + r) * math.Exp(-r) * (R*math.Cosh(R) - math.Sinh(R)) / r / r
		}
	}
	for _, r_i := range r {
		rez = append(rez, rezf(r_i))
	}
	return rez
}

//================================================================================
func rk_method(r, y, dy, dr float64, f, g Diffunc) (float64, float64) {
	/*
	   Численное решение методом явного Рунге-Кутты 4го порядка
	   y=y+sum(k) dy=dy+sum(q)
	*/
	n := 4
	k := make([]float64, n)
	q := make([]float64, n)

	q[0] = f(r, y, dy)
	k[0] = g(r, y, dy)

	q[1] = f(r+dr/2.0, y+dr*k[0]/2.0, dy+q[0]*dr/2.0)
	k[1] = g(r+dr/2.0, y+dr*k[0]/2.0, dy+q[0]*dr/2.0)

	q[2] = f(r+dr/2.0, y+dr*k[1]/2.0, dy+q[1]*dr/2.0)
	k[2] = g(r+dr/2.0, y+dr*k[1]/2.0, dy+q[1]*dr/2.0)

	q[3] = f(r+dr, y+dr*k[2], dy+q[2]*dr)
	k[3] = g(r+dr, y+dr*k[2], dy+q[2]*dr)

	return dr * (k[0] + 2.0*k[1] + 2.0*k[2] + k[3]) / 6.0, dr * (q[0] + 2.0*q[1] + 2.0*q[2] + q[3]) / 6.0
}

func n_rk_method(r, y, dy, dr float64, f, g Diffunc) (float64, float64) {
	/*
	   Численное решение методом неявного Рунге-Кутты
	*/
	return dr * (g(r, y, dy) + g(r+dr, y+dr*f(r, y, dy), dy+dr*g(r, y, dy))) / 2, dr * (f(r, y, dy) + f(r+dr, y+dr*f(r, y, dy), dy+dr*g(r, y, dy))) / 2
}

//==============================================================================================

func make_local(R float64) float64 {
	R = R / l_Deb * 1e-9
	return R
}

func recover(r, R, psi, psi_analit float64) (float64, float64, float64, float64) {
	psi = psi * 2 * n_i
	psi_analit = psi_analit * 2 * n_i
	R = R * l_Deb * 1e9
	r = r * l_Deb * 1e9
	return r, R, psi, psi_analit
}

func arange(start, stop, step float64) []float64 {
	N := int(math.Ceil((stop - start) / step))
	rnge := make([]float64, N, N)
	i := 0
	for x := start; x < stop; x += step {
		rnge[i] = x
		i += 1
	}
	return rnge
}

package num_eu


func Solver_PB(R1, R2, dr, psi_0, rho1, rho2 float64, non_lin, graf, MD_data, method int){
    /*
        Главная функция, решающая нелинеаризованное ур-е Пуассона-Больцмана
        R1, R2:     радиусы: внутренний и внешний
        MD_data:    номер результата МД моделирования || -1 если таковой отсутствует
        method:     0-явный метод РК, 1-неявный метод РК
        dr:         шаг функции
        psi_0:      начальное значение psi для численного метода
        rho1, rho1: плотность заряда внутреняя и внешняя
        non_lin:    1 = линейная, 0 = нелинейная функция солвера
        graf:       вывод графика
    */
    
    R1 = make_local(R1)
    R2 = make_local(R2)
    r = arange(R1*0.6, R1*1.2, dr)
    
	if method{
		solver := n_rk_method
	}else{
		solver := rk_method
	}
    // инициализация psi[-] и dpsi[-]

    psi := []float64{psi_0}
	dpsi := []float64{0.0}
    //  -------------------------------------------------------------------------- 
    Charge_density := func(r, psi float64, non_lin int) (float64){
        /*
                Уравнения Пуассона-Больцмана со ступенькой Хевисайда
        */
        HS := func(r,R, x0) float64{
            /* Функция(ступенька) Хевисайда*/
            return 0.5 * (math.Sign(r+x0) + 1.0) - 0.5* (math.Sign(-R+r+x0) + 1.0)
		}

        if non_lin{
            return (math.Sinh(psi) - rho1*HS(r,R1, 0.05))
		}else{
			switch r<=R1{
			case 1:
				return psi-rho1
			default:
				return psi

			}
		}
	}
    var Diffunc f = func(r, psi, dpsi float64) float64{
        /*
        U(x,y,z) - функция диф. уравнения
        
        */ 
        return (-2.0*dpsi/r + Charge_density(r,psi))
	}
    var Diffunc g = func (r, psi, dpsi float64) float64{
        /*
        ==================================
        V(x,y,z)- функция диф. уравнения
        ==================================
        */
        return dpsi
	}
    //  -------------------------------------------------------------------------- 
    for i:=1; i<len(r); i++{
        k, q := solver(r[i], psi[len(psi)-1], dpsi[len(dpsi)-1], dr, f, g)    // численный шаг    
        psi = append(psi, psi[len(psi)-1]+k) 
        dpsi = append(dpsi, dpsi[len(dpsi)-1]+q)
	}
    //  --------------------------------------------------------------------------
    if graf{
        if non_lin{
            fmt.Println("Нелинейная задача")
		}else{
            print("Линейная задача")
		}
        gr := plt.plot
        psi_analit := pot_DH(r, rho1, R1)

        print("Обезразмеренные величины")
        gr(r, psi_analit, label="Аналитическое")
        gr(r, psi, label="Численное")
        print("Стартовая аналитическая точка: ", psi_analit[0])
        print("Стартовая численная точка: ", psi[0])
        plt.vlines(R1, 0, max(psi_analit)/3)
        plt.title("Явный метод Рунге-Кутты 4  R="+str(R1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        plt.xlabel("r")
        plt.ylabel('\u03C8')
        plt.ylim(0, rho1*1.1)
        plt.show()
        //  --------------------------------------------------------------------------
        if MD_data!= -1{
            print("Исходные величины")
            data, fname := Read_MD_data()
            r, R1, psi, psi_analit = recover(r, R1, psi, psi_analit)
            gr(data[MD_data][::,0], data[MD_data][::,1], label="N_i = "+ fname[MD_data].split('_')[2])
            gr(r, psi_analit, label="Аналитическое")
            gr(r, psi, label="Численное")
            plt.ylim(0, np.max(psi_analit)*1.1)
            plt.title("Явный метод Рунге-Кутты 4  R="+str(R1))
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
            plt.xlabel("r")
            plt.ylabel("N_e")
            plt.show()
		}
	}

}
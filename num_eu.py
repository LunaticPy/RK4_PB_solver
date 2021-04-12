import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os

# Блок 1
#==============================================================================================
#   Константы
d_0         = 0.005                          # [-] -> [nm]          обход нуля
dr          = 1e-4                           # [-] -> [nm]          шаг
nx          = 32000                          # [-]                  число точек разбиения
e           = -1.602*1e-19                  # [A*s]                заряд электрона
k_Bolt      = 1.38*1e-23                    # [J/K]               постоянная Больцмана
eps0        = 8.85418781762039*1e-12        # [F*м^−1]             Диэлектрическая проницаемость вакуума
one_eV      = 1.602176620898*1e-19          # [J]                  Один электронвольт

#слишком маленькие числа для функций numpy
#   Входные данные
temp        = 5.0                            # [eV]                 температура
R           = 3.0                            # [nm]                 радиус ионной сферы
n_i         = 27.6                           # [nm−3]               плотность электронов: плотность ионов, умноженная на среднюю степень ионизации
#==============================================================================================

# Входные данные в СИ

temp_SI     = temp*one_eV/k_Bolt             # [K]                 температура
R_SI        = R*1e-9                         # [m]                 радиус ионной сферы
n_i_SI      = n_i*1e27                       # [m−3]               плотность электронов: плотность ионов, умноженная на среднюю степень ионизации

#==============================================================================================

# Переменные внутри главной функции в СИ

l_Berum   = e*e/(4*np.pi*eps0*k_Bolt*temp_SI)              # [m]                 длина Бьеррума/Ландау
l_Deb     = np.sqrt((eps0*k_Bolt*temp_SI)/(e*e*n_i_SI))    # [m]                 длина Дебая

#==============================================================================================
# Внутренние единицы
rho1        = 0.5                            # [обзразмеренно на l_Deb-3]       концентрация(плотность) заряда ионов внутреняя(обезразмерянная и деленная на 2)
rho2        = 0                              # [обзразмеренно на l_Deb-3]       концентрация(плотность) заряда ионов внешняя(вакуум)
temp        = 5                              # [] нет нужды
R1, R2      = R_SI/l_Deb, R_SI/l_Deb         # [обзразмеренно на l_Deb]          радиус ионной сферы


#==============================================================================================
#==============================================================================================



def read_MD_data(mol_d = r"C:\\Users\\1\\Downloads\\densprof_coul_5eV\\densprof_coul_5eV\\"):
    """ 
    Чтение данных молекулярной динамики
    mol_d - путь к МД результатам 
    data - [list] результат
    """
    data = []
    fname = []

    for i in next(os.walk(mol_d))[2]:
        if i[-3:]=='dat':
            fname.append(i[:-4])
            data.append(np.loadtxt(mol_d+i))
    return data, fname

#==============================================================================================
#   функции на граничные условия
def phi_0_DH(rho,R):
    """
         Внутренняя область аналитического решения
        -Обезразмерянное
    """
    return -rho*(1+R)
def phi_s_DH(rho,R):
    """ 
        Внешняя область аналитического решения
        -Обезразмерянное
    """
    return -rho/R*(np.sinh(R)-R*np.cosh(R))

#   расширение функций на граничные условия
def pot_DH(r,rho,R):
    """
         Полное аналитическое решение
        -Обезразмерянное
    """
    return np.piecewise(r,[np.abs(r)<R,np.abs(r)>=R],[lambda r: np.sinh(r)/r*(phi_0_DH(rho,R))+rho*np.exp(R),lambda r: R*phi_s_DH(rho,R)*np.exp(R-r)/r])*np.exp(-R)
def dpot_DH(r,rho,R):
    """ 
        Производная полного аналитического решения
        -Обезразмерянное
    """
    return np.piecewise(r,[np.abs(r)<R,np.abs(r)>=R],[lambda r: -rho*(1+R)*np.exp(-R)*(r*np.cosh(r)-np.sinh(r))/r**2,lambda r: -rho*(1+r)*np.exp(-r)*(R*np.cosh(R)-np.sinh(R))/r**2 ])
#==============================================================================================

def rk_method (r, y, dy, dr, f, g):
    """ 
        Численное решение методом явного Рунге-Кутты 4го порядка
        y=y+sum(k) dy=dy+sum(q)
    """     
    n = 4
    k, q = np.zeros(n), np.zeros(n)     
    
    q[0] = f(r, y, dy)
    k[0] = g(r, y, dy)

    q[1] = f(r+dr/2.0, y+dr*k[0]/2.0, dy+q[0]*dr/2.0)
    k[1] = g(r+dr/2.0, y+dr*k[0]/2.0, dy+q[0]*dr/2.0)

    q[2] = f(r+dr/2.0, y+dr*k[1]/2.0, dy+q[1]*dr/2.0)
    k[2] = g(r+dr/2.0, y+dr*k[1]/2.0, dy+q[1]*dr/2.0)

    q[3] = f(r+dr, y+dr*k[2], dy+q[2]*dr)
    k[3] = g(r+dr, y+dr*k[2], dy+q[2]*dr)

    return  dr*(k[0]+2.0*k[1]+2.0*k[2]+k[3])/6.0, dr*(q[0]+2.0*q[1]+2.0*q[2]+q[3])/6.0

def n_rk_method (r, y, dy, dr, f, g):
    """ 
        Численное решение методом неявного Рунге-Кутты
    """ 
    return  dr*(g(r, y, dy)+g(r+dr, y+dr*f(r, y, dy), dy+dr*g(r, y, dy)))/2, dr*(f(r, y, dy)+f(r+dr, y+dr*f(r, y, dy), dy+dr*g(r, y, dy)))/2

#==============================================================================================

def make_local(R):
    R = R/l_Deb*1e-9
    return R


def recover(r, R, psi, psi_analit):
    psi = psi*2*n_i
    psi_analit = psi_analit*2*n_i
    R = R*l_Deb*1e9
    r = r*l_Deb*1e9
    return r, R, psi, psi_analit

def lap(r, psi):
    dr = r[1]-r[0]
    dpsi  = np.gradient(psi, dr)
    ddpsi = np.gradient(dpsi, dr)
    return ddpsi+2*dpsi/r

def recover_nonlin(r, R, psi, psi_analit, R1):
    psi = lap(r, psi)
    dr = r[1]-r[0]
    psi[r<R1-dr*7]+=rho1
    dpsi = np.abs(np.gradient(psi, r[1]-r[0]))
    psi = psi*2.0*n_i
    psi_analit = psi_analit*2.0*n_i
    R, r = R*l_Deb*1e9, r*l_Deb*1e9
    
    l = dpsi<1
    r, psi, psi_analit = r[l], psi[l], psi_analit[l]
    return r, R, psi, psi_analit

#==============================================================================================


def Solver_PB(R1=R1, R2=R2, MD_data=-1, method=0, dr=dr, psi_0=0.0, dpsi_0=0.0, rho1=rho1, rho2=rho2, non_lin=1, graf=1):
    """
        Главная функция, решающая нелинеаризованное ур-е Пуассона-Больцмана
        R1, R2:     радиусы: внутренний и внешний
        MD_data:    номер результата МД моделирования || -1 если таковой отсутствует
        method:     0-явный метод РК, 1-неявный метод РК
        dr:         шаг функции
        psi_0:      начальное значение psi для численного метода
        rho1, rho1: плотность заряда внутреняя и внешняя
        non_lin:    0 = линейная, 1 = нелинейная функция солвера
        graf:       вывод графика
    """
    R1 = make_local(R1)
    R2 = make_local(R2)
    r = np.arange(R1*0.6, R1*1.2, dr)
    
    solver = n_rk_method if method else rk_method
    # инициализация psi[-] и dpsi[-]
    psi, dpsi = np.array([psi_0]), np.array([dpsi_0])
    ####  -------------------------------------------------------------------------- 
    def Charge_density(r, psi, d_0=d_0, R1=R1, R2=R2, rho1=rho1, rho2=rho2):
        """
                Уравнения Пуассона-Больцмана со ступенькой Хевисайда
        """
        def HS(r,R, x0):
            """ Функция(ступенька) Хевисайда"""
            return 0.5 * (np.sign(r+x0) + 1.0) - 0.5* (np.sign(-R+r+x0) + 1.0)
        
        if non_lin:
            return (np.sinh(psi) - rho1*HS(r,R1, 0.05))
        else:
            return np.piecewise(r, r<=R1,[lambda r, psi: psi-rho1,lambda r, psi: psi], psi)

    def f(r, psi, dpsi):
        """
        U(x,y,z) - функция диф. уравнения
        
        """ 
        return (-2.0*dpsi/r + Charge_density(r,psi))
    
    def g(r, psi, dpsi):
        """
        ==================================
        V(x,y,z)- функция диф. уравнения
        ==================================
        """
        return dpsi
   
    ####  -------------------------------------------------------------------------- 
    for i in range(1, r.shape[0]):
        k, q = solver(r[i], psi[-1], dpsi[-1], dr, f, g)    # численный шаг    
        psi = np.append(psi, psi[-1]+k) 
        dpsi = np.append(dpsi, dpsi[-1]+q)
    ####  --------------------------------------------------------------------------
    if graf:
        if non_lin:
            print("Нелинейная задача")
        else:
            print("Линейная задача")
        sp = 0 # отступ
        gr = plt.plot
        psi_analit = pot_DH(r, rho1, R1)
        print("Обезразмеренные величины")
        gr(r[sp:], psi_analit[sp:], label="Аналитическое")
        gr(r[sp:], psi[sp:], label="Численное")
        plt.vlines(R1, 0, max(psi_analit[sp:])/3)
        plt.title("Явный метод Рунге-Кутты 4  R="+str(R1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel("r")
        plt.ylabel('\u03C8')
        plt.ylim(0, rho1*1.1)
        plt.show()
        ####  --------------------------------------------------------------------------
        if MD_data!= -1:
            print("Исходные величины в плотности электронов")
            data, fname = read_MD_data()
            if non_lin: rec = recover_nonlin 
            else: rec = recover
            r, R1, psi, psi_analit = rec(r, R1, psi, psi_analit, R1)
            gr(data[MD_data][::,0], data[MD_data][::,1], label='N_i = '+ fname[MD_data].split('_')[2])
            gr(r[sp:], psi_analit[sp:], label="Аналитическое")
            gr(r[sp:], psi[sp:], label="Численное")
            plt.ylim(0, np.max(psi_analit)*1.1)
            plt.title("Явный метод Рунге-Кутты 4  R="+str(R1))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.xlabel("r")
            plt.ylabel("N_e")
            plt.show()
    else:        
        if  np.isnan(psi[-1]):
    #         print(np.sum(psi<-0.1)>r.shape[0]*0.1)
            if np.sum(psi<0)>1:
                return -100
            else:
                return 100
        return psi[-1]
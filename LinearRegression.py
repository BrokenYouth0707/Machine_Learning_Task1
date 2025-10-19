import DataPoints
import matplotlib.pyplot as plt
# hypothese
# @x should be a row.
def h(x,th):
    assert len(x) == len(th) == par_number + 1
  
    result = 0
    for i in range(0, par_number + 1):
        result += x[i] * th[i]
    return result

def cost_func(th):
    result = 0 
    for i in range(0, sample_number):
        result += (h(sample_x[i],th) - sample_y[i]) * (h(sample_x[i],th) - sample_y[i])
    return 1/2 * result

# Take the derivative of J(cost_func) with theta_i.
# In our situation, i is in the set {0,1}.
def take_p_deriv(point, i):
   assert i == 1 or i == 0
   p_deriv = 0
   for j in range(0, sample_number):
       p_deriv += (h(sample_x[j],point) - sample_y[j]) * sample_x[j][i]
   return p_deriv

def take_grad(point):
    assert len(point) == len(sample_x[0]) 

    grad = [0.0] * len(point)
    for i in range(0, par_number + 1):
        grad[i] = take_p_deriv(point,i)
    return grad
    

def run_grad_des_once(point, study_rate):

    p = [0.0] * len(point)
    grad = take_grad(point)
    for i in range(0, par_number + 1):
       p[i] = point[i] - study_rate * grad[i] 
    return p

def run_grad_des(point, study_rate):
    P_prev = point
    P_cur  = [0.0] * len(point)
    J_prev = 0
    i = 0
    while True:
        assert P_prev != None

        P_cur = run_grad_des_once(P_prev, study_rate) 
        J_cur = cost_func(P_cur)

        # Record theta to plot pictures
        if i < 3:
            th_rec.append(P_cur)

        if abs(J_prev - J_cur) < epsilon:
            return P_cur
        P_prev = P_cur
        J_prev = J_cur
        i+=1

def plot_pic(P):
    x_min = min(DataPoints.x)
    x_max = max(DataPoints.x)
    
    x_line = [x_min + i*(x_max - x_min)/100 for i in range(101)]
    y_line = [h([1,xi], P) for xi in x_line]

    plt.scatter(DataPoints.x, sample_y, color='blue', label='Data Points')   
    plt.plot(x_line, y_line, color='red', label='Regression Line')   
 #   plt.xlabel("x")
 #   plt.ylabel("y")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1)) 
    plt.gca().set_xlim(-2, 2) 
    plt.gca().set_ylim(-2, 2)
    plt.title("Linear Regression")

    plt.legend()
    plt.show()
 
sample_x = []
sample_y = []
sample_number  = 0
par_number     = 0
epsilon        = 0

# Record theta after first, second and third iterations.
th_rec = []

def main():
    global sample_x
    global sample_y
    global sample_number 
    global par_number
    global epsilon
    # In our situtation, the structure of sample_x just likes this:
    # 1 data_1
    # 1 data_2
    # 1 data_3
    # ...
    # ...
    # 1 data_n
    # Hence,it is a array of n by 2,that is, samlpe_x[n][2]
    sample_x      = [[1,a] for a in DataPoints.x]
    
    # samole_y is a vector, it should be n.
    sample_y      = DataPoints.y
    
    # this number should be n.
    sample_number =  len(sample_x)
    
    #In our situation, it is 1. 
    par_number    =  len(sample_x[0]) - 1 
    
    epsilon       = 1e-15
    init_point    = [10,10] 
    P = run_grad_des(init_point, 0.02)
    plot_pic(th_rec[0])
    plot_pic(th_rec[1])
    plot_pic(th_rec[2])
    plot_pic(P) 
if __name__ == "__main__":
    main()


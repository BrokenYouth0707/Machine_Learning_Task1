
# Sample_x should be a matrix. However,it is a line here.
sample_x      = 
sample_y      =
sample_number =  sample.size()
par_number    = 
epsilon       = 1e - 6
init_point    = 

#hypothese
def h(x,th):


def cost_func():

# To take the derivative of J with theta_i.
# In our situation, i is in the set {0,1}.
def take_p_deriv(point, i):
   p_deriv = 0
   for j in range(sample_number):
       p += (h(x[j],point) - sample_y[j]) * sample_x[j]

def take_grad(point):
    for i in range(0, par_number + 1):
        grad[i] = take_p_deriv(point,i)
    return grad
    

def run_grad_des_once(point, study_rate):


def run_grad_des(point, study_rate):
     
    while True:
        P_cur = run_grad_des_once(####) 
        J_cur = cost_func(P_cur)
        if abs(J_prev - J_cur) < epsilon:
            return p_cur
        p_prev = p_cur
        J_prev = J_cur


def main():




if __name__ == "__main__":
    main()


import numpy as np 
from scipy.integrate import ode
import matplotlib.pyplot as plt 
"""
Exp data as boundary conditions. 
"""
def pbpk_ode(t, y, 
             kin_li, kout_li, kon_li, ke_li, kcl_li,
             kin_s, kout_s, kon_s, ke_s, kcl_s,
             kin_gi, kout_gi, kon_gi, ke_gi, kcl_gi,
             kin_k, kout_k, kon_k, ke_k, kcl_k,
             kin_h, kout_h, kon_h, ke_h, kcl_h,
             kin_lu, kout_lu, kon_lu, ke_lu, kcl_lu, 
             kin_m, kout_m, kon_m, ke_m, kcl_m,
             kon_mono, ke_mono, kcl_mono):
    """
    
    """
    sum_kin, sum_kout_m = 0, 0 

    sum_kin += kin_li + kin_s + kin_gi + kin_k +kin_h + kin_lu + kin_m
    sum_kout_m += kout_li*y[1]+kout_s*y[2]+kout_gi*y[3]+kout_k*y[4]+kout_h*y[5]+kout_lu*y[6]+kout_m*y[7]

    dp_dt = -sum_kin*y[0] + sum_kout_m - kon_mono*y[0]*(1-y[8])*10**-8
    dmono_dt = kon_mono*y[0]*(1-y[8])*10**-8 - ke_mono*y[8]

    dli_dt = kin_li*y[0] - kout_li*y[1] - kon_li*y[1]*(1-y[9])*10**-8
    dli_cell_dt = kon_li*y[1]*(1-y[9])*10**-8 - ke_li*y[9]

    ds_dt = kin_s * y[0] - kout_s * y[2] - kon_s*y[2]*(1-y[10])*10**-8
    ds_cell_dt = kon_s*y[2]*(1-y[10])*10**-8 - ke_s * y[10]

    dgi_dt = kin_gi * y[0] - kout_gi * y[3] - kon_gi *y[3]*(1-y[11])*10**-8
    dgi_cell_dt = kon_gi *y[3]*(1-y[11])*10**-8 - ke_gi * y[11]

    dk_dt = kin_k * y[0] - kout_k * y[4] - kon_k*y[4]*(1-y[12])*10**-8 
    dk_cell_dt = kon_k*y[4]*(1-y[12])*10**-8 - ke_k * y[12]

    dh_dt = kin_h * y[0] - kout_h * y[5] - kon_h*y[5]*(1-y[13])*10**-8
    dh_cell_dt = kon_h*y[5]*(1-y[13])*10**-8 - ke_h * y[13]

    dlu_dt = kin_lu * y[0] - kout_lu * y[6] - kon_lu*y[6]*(1-y[14])*10**-8
    dlu_cell_dt = kon_lu*y[6]*(1-y[14])*10**-8 - ke_lu * y[14]
    
    dm_dt = kin_m * y[0] - kout_m * y[7] - kon_m*y[7]*(1-y[15])*10**-8
    dm_cell_dt = kon_m*y[7]*(1-y[15])*10**-8 - ke_m * y[15]

    return [dp_dt, dli_dt, ds_dt, dgi_dt, dk_dt, dh_dt, dlu_dt, dm_dt,
            dmono_dt, dli_cell_dt, ds_cell_dt, dgi_cell_dt, dk_cell_dt, 
            dh_cell_dt, dlu_cell_dt, dm_cell_dt]

if __name__ == "__main__":
    # Initialize the solver
    r = ode(pbpk_ode).set_integrator('zvode', method='bdf', order=15)
    org_cells = ["Plasma", "Liver", "Spleen", "GI", "Kidney", "Heart", "Lungs", 
            "Muscle", "Monocyte", "LiverCell", "SpleenCell", "GICell", 
            "KidneyCell", "HeartCell", "LungsCell", "MuscleCell"]
    # Set initial conditions 
    y0 = [1.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0]
    t0 = 0.0  # Initial time and parameters
    # Integrate step-by-step
    t1 = 100
    dt = 1
    # time_points = np.arange(t0, t1, dt)
    time_points = [10, 60, 120, 240]
    solution = []

    # Net work. 
    # function set the parameter 
    for i in range(1):
        # Change with different parameters. 
        para = (
            0.008, 0.01, 3.43*10**5, 10**-4, 10**-4, # Liver data. 
            0.006, 0.01, 3.43*10**6, 1.2*10**-4, 10**-4,  # Spleen data. 
            0.004, 0.01, 3.43*10**6, 1.2*10**-4, 10**-4,  # GI data. 
            0.001, 0.01, 3.43*10**5, 10**-4, 10**-4,  # Kidney data. 
            0.0015, 0.002, 3.43*10**5, 10**-4, 10**-4,  # Heart data. 
            0.004, 0.01, 3.43*10**5, 10**-4, 10**-4,  # Lung data. 
            0.0015, 0.002, 3.43*10**5, 10**-4, 10**-4,  # Kidney data. 
            6.15*10**7, 10**-4, 10**-4, # Monocyte data.
                )
        r.set_initial_value(y0, t0).set_f_params(*para) 
        for t in time_points:
            if r.successful() and r.t < t:
                r.integrate(t)
            solution.append(r.y)
        # Convert solution to a numpy array for easier handling
        solution = np.array(solution)
        print(solution)
        plt.figure()
        plt.plot(time_points, solution)
        legend = ["Plasma", "Liver", "Spleen", "GI", "Kidney", "Heart", "Lungs", 
                  "Muscle", "Monocyte", "LiverCell", "SpleenCell", "GICell", 
                  "KidneyCell", "HeartCell", "LungsCell","MuscleCell"]

        plt.legend(legend, ncol=3)
        plt.show()

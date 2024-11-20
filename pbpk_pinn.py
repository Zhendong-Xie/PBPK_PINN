import numpy as np 
from scipy.integrate import ode
import matplotlib.pyplot as plt 
"""
Exp data as boundary conditions. 
"""
def pbpk_ode(t, y, 
             c_mono_R, kon_mono, ke_mono, kr_mono, # kcl_mono,
             kin_li, kout_li, 
             c_li_R, kon_li, ke_li, kr_li, # kcl_li,
             kin_s, kout_s, 
             c_s_R, kon_s, ke_s, kr_s, # kcl_s,
             kin_gi, kout_gi, 
             c_gi_R, kon_gi, ke_gi, kr_gi, # kcl_gi,
             kin_k, kout_k, 
             c_k_R, kon_k, ke_k, kr_k, # kcl_k,
             kin_h, kout_h, 
             c_h_R, kon_h, ke_h, kr_h, # kcl_h,
             kin_lu, kout_lu, 
             c_lu_R, kon_lu, ke_lu, kr_lu, # kcl_lu, 
             kin_m, kout_m, 
             c_m_R, kon_m, ke_m, kr_m, # kcl_m
             ):
    """
    
    """
    sum_kin, sum_kout_m = 0, 0  
    if t < 200: # t< 200s 
        sum_kin += kin_li + kin_s + kin_gi + kin_k +kin_h + kin_lu + kin_m
        sum_kout_m += kout_li*y[1]+kout_s*y[2]+kout_gi*y[3]+kout_k*y[4]+kout_h*y[5]+kout_lu*y[6]+kout_m*y[7]

        dp_dt = -sum_kin*y[0]+sum_kout_m - kon_mono*y[0]*c_mono_R
        dmono_B_dt = kon_mono*y[0]*c_mono_R - ke_mono*y[8]
        c_mono_R += (-kon_mono*y[0]*c_mono_R)

        dli_dt = kin_li*y[0] - kout_li*y[1] - kon_li*y[1]*c_li_R
        dli_cell_dt = kon_li*y[1]*c_li_R - ke_li*y[9]
        c_li_R += (-kon_li*y[1]*c_li_R)

        ds_dt = kin_s * y[0] - kout_s * y[2] - kon_s*y[2]*c_s_R
        ds_cell_dt = kon_s*y[2]*c_s_R- ke_s * y[10]
        c_s_R += (-kon_s*y[2]*c_s_R)

        dgi_dt = kin_gi * y[0] - kout_gi * y[3] - kon_gi *y[3]*c_gi_R
        dgi_cell_dt = kon_gi *y[3]*c_gi_R -  ke_gi * y[11]
        c_gi_R += (-kon_gi *y[3]*c_gi_R)

        dk_dt = kin_k * y[0] - kout_k * y[4] - kon_k*y[4]*c_k_R
        dk_cell_dt = kon_k*y[4]*c_k_R- ke_k * y[12]
        c_k_R += (-kon_k*y[4]*c_k_R)

        dh_dt = kin_h * y[0] - kout_h * y[5] - kon_h*y[5]*(c_h_R-y[13])
        dh_cell_dt = kon_h*y[5]*(c_h_R-y[13])- ke_h * y[13]
        c_h_R += (-kon_h*y[5]*c_h_R)

        dlu_dt = kin_lu * y[0] - kout_lu * y[6] - kon_lu*y[6]*c_lu_R
        dlu_cell_dt = kon_lu*y[6]*c_lu_R- ke_lu * y[14]
        c_lu_R += (-kon_lu*y[6]*c_lu_R)
        
        dm_dt = kin_m * y[0] - kout_m * y[7] - kon_m*y[7]*(c_m_R-y[15])
        dm_cell_dt = kon_m*y[7]*(c_m_R-y[15])- ke_m * y[15]
        c_m_R += (-kon_m*y[7]*c_m_R)
    
    else:
        sum_kin += kin_li + kin_s + kin_gi + kin_k +kin_h + kin_lu + kin_m
        sum_kout_m += kout_li*y[1]+kout_s*y[2]+kout_gi*y[3]+kout_k*y[4]+kout_h*y[5]+kout_lu*y[6]+kout_m*y[7]

        dp_dt = -sum_kin*y[0]+sum_kout_m - kon_mono*y[0]*c_mono_R
        dmono_B_dt = kon_mono*y[0]*c_mono_R - ke_mono*y[8]
        c_mono_R += (-kon_mono*y[0]*c_mono_R + kr_mono * c_mono_R)

        dli_dt = kin_li*y[0] - kout_li*y[1] - kon_li*y[1]*c_li_R
        dli_cell_dt = kon_li*y[1]*c_li_R - ke_li*y[9]
        c_li_R += (-kon_li*y[1]*c_li_R + kr_li * c_li_R)

        ds_dt = kin_s * y[0] - kout_s * y[2] - kon_s*y[2]*c_s_R
        ds_cell_dt = kon_s*y[2]*c_s_R- ke_s * y[10]
        c_s_R += (-kon_s*y[2]*c_s_R + kr_s*c_s_R)

        dgi_dt = kin_gi * y[0] - kout_gi * y[3] - kon_gi *y[3]*c_gi_R
        dgi_cell_dt = kon_gi *y[3]*c_gi_R -  ke_gi * y[11]
        c_gi_R += (-kon_gi *y[3]*c_gi_R + kr_gi * c_gi_R)

        dk_dt = kin_k * y[0] - kout_k * y[4] - kon_k*y[4]*c_k_R
        dk_cell_dt = kon_k*y[4]*c_k_R- ke_k * y[12]
        c_k_R += (-kon_k*y[4]*c_k_R + c_k_R * kr_k)

        dh_dt = kin_h * y[0] - kout_h * y[5] - kon_h*y[5]*(c_h_R-y[13])
        dh_cell_dt = kon_h*y[5]*(c_h_R-y[13])- ke_h * y[13]
        c_h_R += (-kon_h*y[5]*c_h_R + c_h_R*kr_h)

        dlu_dt = kin_lu * y[0] - kout_lu * y[6] - kon_lu*y[6]*c_lu_R
        dlu_cell_dt = kon_lu*y[6]*c_lu_R- ke_lu * y[14]
        c_lu_R += (-kon_lu*y[6]*c_lu_R + c_lu_R * kr_lu)
        
        dm_dt = kin_m * y[0] - kout_m * y[7] - kon_m*y[7]*(c_m_R-y[15])
        dm_cell_dt = kon_m*y[7]*(c_m_R-y[15])- ke_m * y[15]
        c_m_R += (-kon_m*y[7]*c_m_R + kr_m * c_m_R)

    return [dp_dt, dli_dt, ds_dt, dgi_dt, dk_dt, dh_dt, dlu_dt, dm_dt, # 0-7
            dmono_B_dt, dli_cell_dt, ds_cell_dt, dgi_cell_dt, dk_cell_dt, 
            dh_cell_dt, dlu_cell_dt, dm_cell_dt # 8-15
            ]

if __name__ == "__main__":
    # Initialize the solver
    r = ode(pbpk_ode).set_integrator('zvode', method='bdf', order=15)
    org_cells = ["Plasma", "Liver", "Spleen", "GI", "Kidney", "Heart", "Lungs", 
            "Muscle", "Monocyte", "LiverCell", "SpleenCell", "GICell", 
            "KidneyCell", "HeartCell", "LungsCell", "MuscleCell"]
    # Set initial conditions 
    y0 = [1.0*10**-5, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0]
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
            1.15*10**-8, 6.15*10**7, 10**-4, 10**-4, # Monocyte data.
            0.008, 0.01, 
            1.43*10**-5, 3.43*10**5, 10**-4, 10**-4, # Liver data. 
            0.006, 0.01, 
            1.43*10**-6, 3.43*10**6, 1.2*10**-4, 10**-4,  # Spleen data. 
            0.004, 0.01, 
            1.43*10**-6, 3.43*10**6, 1.2*10**-4, 10**-4,  # GI data. 
            0.001, 0.01, 
            1.43*10**-5, 3.43*10**5, 10**-4, 10**-4,  # Kidney data. 
            0.0015, 0.002, 
            1.43*10**-5, 3.43*10**5, 10**-4, 10**-4,  # Heart data. 
            0.004, 0.01, 
            1.43*10**-5, 3.43*10**5, 10**-4, 10**-4,  # Lung data. 
            0.0015, 0.002, 
            1.43*10**-5, 3.43*10**5, 10**-4, 10**-4,  # Kidney data. 
                )
        r.set_initial_value(y0, t0).set_f_params(*para) 
        for t in time_points:
            if r.successful() and r.t < t:
                r.integrate(t)
            solution.append(r.y)
        # Convert solution to a numpy array for easier handling
        solution = np.array(solution)

        plt.figure()
        plt.plot(time_points, solution)
        legend = ["Plasma", "Liver", "Spleen", "GI", "Kidney", "Heart", "Lungs", 
                  "Muscle", "Monocyte", "LiverCell", "SpleenCell", "GICell", 
                  "KidneyCell", "HeartCell", "LungsCell","MuscleCell"]

        plt.legend(legend, ncol=3)
        plt.show()

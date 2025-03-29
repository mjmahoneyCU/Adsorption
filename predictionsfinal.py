import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, simpson
import pandas as pd

# Page setup
st.set_page_config(page_title="Fixed-Bed Adsorption Simulator", layout="wide")
st.title("🧪 Fixed-Bed Adsorption Column Simulator")
st.write("""
Explore breakthrough behavior in a fixed-bed column with Langmuir adsorption.
Compare up to 5 simulations with different operating conditions.
""")

# --- Global Column Properties ---
st.sidebar.header("Global Column Settings")
column_length = st.sidebar.slider("Column Length (cm)", 5.0, 20.0, 10.0)
porosity = st.sidebar.slider("Bed Porosity", 0.2, 0.9, 0.4)
column_diameter = st.sidebar.slider("Column Diameter (cm)", 0.5, 2.0, 0.7)
sim_time = st.sidebar.slider("Simulation Duration (min)", 20, 500, 100)
nz = st.sidebar.slider("Number of Spatial Points", 10, 200, 50)

# --- Number of Simulations ---
num_sims = st.sidebar.slider("Number of Simulations to Compare", 1, 5, 1)

# --- Derived geometry ---
area = np.pi * (column_diameter / 2) ** 2
column_volume = column_length * area  # in cm³ = mL
z = np.linspace(0, column_length, nz)
dz = z[1] - z[0]
t_eval = np.linspace(0, sim_time, 300)

# Initial conditions
def initial_conditions(nz, c0):
    c_init = np.zeros(nz)
    c_init[0] = c0 * 0.001  # small initial pulse
    return np.concatenate([c_init, np.zeros(nz)])

# Model function
def make_model(v, DL, Ka, KL, qmax, c0):
    def model(t, y):
        c = y[:nz]
        q = y[nz:]
        dc_dt = np.zeros(nz)
        dq_dt = Ka * ((qmax * KL * c) / (1 + KL * c) - q)

        for i in range(1, nz - 1):
            conv = -v * (c[i] - c[i - 1]) / dz
            disp = DL * (c[i + 1] - 2 * c[i] + c[i - 1]) / dz**2
            dc_dt[i] = conv + disp - ((1 - porosity) / porosity) * dq_dt[i]

        dc_dt[0] = (-v * (c[1] - c[0]) / dz + DL * (c[1] - c[0]) / dz**2 +
                    v * (c0 - c[0]) / dz - ((1 - porosity) / porosity) * dq_dt[0])
        dc_dt[-1] = DL * (c[-2] - c[-1]) / dz**2 - v * (c[-1] - c[-2]) / dz - ((1 - porosity) / porosity) * dq_dt[-1]

        return np.concatenate([dc_dt, dq_dt])
    return model

# --- Simulation Setups ---
profiles = []
for i in range(num_sims):
    st.sidebar.subheader(f"Simulation {i + 1} Parameters")
    flow_rate = st.sidebar.slider(f"Flow Rate {i+1} (mL/min)", 1.0, 30.0, 10.0)
    DL = st.sidebar.slider(f"Dispersion DL {i+1} (cm²/min)", 0.01, 0.1, 0.05)
    Ka = st.sidebar.slider(f"Adsorption Ka {i+1} (1/min)", 0.1, 10.0, 1.0)
    KL = st.sidebar.slider(
        f"Langmuir KL {i+1} (mL/mg)",
        min_value=0.001,
        max_value=1.0,
        value=0.15,
        step=0.001,
        format="%.3f"
    )
    qmax = st.sidebar.slider(f"qmax {i+1} (mg/mL)", 10.0, 100.0, 65.0)
    c0 = st.sidebar.slider(f"Inlet Concentration c₀ {i+1} (mg/mL)", 0.5, 25.0, 20.0)

    v = flow_rate / area  # superficial velocity
    profiles.append({"flow_rate": flow_rate, "v": v, "DL": DL, "Ka": Ka, "KL": KL, "qmax": qmax, "c0": c0})

# --- Run Simulations ---
if st.button("▶️ Run Simulations"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    summary_data = []

    for i, p in enumerate(profiles):
        with st.spinner(f"Running Simulation {i+1}..."):
            sol = solve_ivp(
                make_model(p['v'], p['DL'], p['Ka'], p['KL'], p['qmax'], p['c0']),
                [0, sim_time],
                initial_conditions(nz, p['c0']),
                t_eval=t_eval,
                method="BDF",
                rtol=1e-5,
                atol=1e-6
            )
            c_out = sol.y[nz - 1, :]
            c_out = np.maximum(c_out, 0)

            ax1.plot(sol.t, c_out, label=f"Sim {i+1}")

            # Mass In = c0 * Q * total time, Area under c_out = total output
            total_input_mass = p['c0'] * p['flow_rate'] * sol.t[-1]  # mg
            total_output_mass = simpson(y=c_out * p['flow_rate'], x=sol.t)  # mg
            mass_bound = total_input_mass - total_output_mass

            # Theoretical max capacity (mg)
            resin_volume = column_volume * (1 - porosity)  # mL
            max_binding_capacity = p['qmax'] * resin_volume  # mg

            dynamic_binding_capacity = mass_bound / column_volume  # mg/mL

            # Find time to 10% breakthrough
            breakthrough_threshold = 0.1 * p['c0']
            idx_bt = np.argmax(c_out >= breakthrough_threshold)
            if idx_bt > 0:
                t_bt = np.interp(breakthrough_threshold, [c_out[idx_bt-1], c_out[idx_bt]], [sol.t[idx_bt-1], sol.t[idx_bt]])
                t_bt_eval = sol.t[sol.t <= t_bt]
                c_bt_eval = np.interp(t_bt_eval, sol.t, c_out)
                mass_in_bt = p['c0'] * p['flow_rate'] * t_bt
                mass_out_bt = simpson(y=c_bt_eval * p['flow_rate'], x=t_bt_eval)
                dbc_at_bt = (mass_in_bt - mass_out_bt) / column_volume
            else:
                dbc_at_bt = np.nan

            summary_data.append({
                "Simulation": f"Sim {i+1}",
                "Flow Rate (mL/min)": p['flow_rate'],
                "DL (cm²/min)": p['DL'],
                "Ka (1/min)": p['Ka'],
                "KL (mL/mg)": p['KL'],
                "qmax (mg/mL)": p['qmax'],
                "c₀ (mg/mL)": p['c0'],
                "Resin Volume (mL)": resin_volume,
                "Max Capacity (mg)": max_binding_capacity,
                "Total Mass In (mg)": total_input_mass,
                "Total Mass Out (mg)": total_output_mass,
                "Final Mass Bound (mg)": mass_bound,
                "Dynamic Binding Capacity (mg/mL)": dynamic_binding_capacity,
                "DBC at 10% Breakthrough (mg/mL)": dbc_at_bt
            })

    ax1.set_title("Breakthrough Curves")
    ax1.set_ylabel("Outlet Concentration (mg/mL)")
    ax1.set_xlabel("Time (min)")
    ax1.grid(True)
    ax1.legend()

    st.pyplot(fig)
    st.success("✅ Simulations complete.")

    # --- Summary Table ---
    st.subheader("📊 Simulation Summary Table")
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.format({
        "Resin Volume (mL)": "{:.2f}",
        "Max Capacity (mg)": "{:.1f}",
        "Total Mass In (mg)": "{:.2f}",
        "Total Mass Out (mg)": "{:.2f}",
        "Final Mass Bound (mg)": "{:.2f}",
        "Dynamic Binding Capacity (mg/mL)": "{:.2f}",
        "DBC at 10% Breakthrough (mg/mL)": "{:.2f}"
    }))

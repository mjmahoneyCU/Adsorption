import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fixed-Bed Adsorption Simulator", layout="wide")
st.title("🧪 Fixed-Bed Adsorption Simulator")
st.write("""
Explore breakthrough behavior in a fixed-bed column with Langmuir adsorption.
Compare up to 5 simulations with different operating conditions.
""")

# --- PARAMETER DESCRIPTIONS ---
with st.expander("ℹ️ Simulation Parameter Descriptions", expanded=True):
    st.markdown("""
    Adjust the sidebar sliders to explore how each parameter affects breakthrough behavior in a fixed-bed system.

    - **Column Length**: Longer columns give more space for solute-resin interactions, potentially delaying breakthrough.
    - **Bed Porosity**: The fraction of the bed volume that is liquid-filled. Lower porosity increases residence time but may cause higher pressure drop.
    - **Column Diameter**: Affects the column cross-sectional area and thus the superficial velocity at fixed flow rate.
    - **Flow Rate**: Determines how fast the feed moves through the bed. Higher flow means shorter contact time.
    - **Dispersion Coefficient (DL)**: Controls spreading of solute in the axial direction. Larger values smear the breakthrough curve.
    - **Ka (Adsorption Rate Constant)**: A lumped parameter describing how fast protein transfers to the resin.
    - **KL (Langmuir Constant)**: Defines how tightly protein binds to resin. Smaller values indicate stronger binding.
    - **qmax**: Maximum binding capacity of the resin (mg protein / mL resin).
    - **c₀**: Inlet protein concentration (mg/mL).
    - **Number of Spatial Points**: Controls resolution of the numerical solution along the column.
    """)

# --- GLOBAL COLUMN SETTINGS ---
st.sidebar.header("Global Column Settings")
column_length = st.sidebar.slider("Column Length (cm)", 5.0, 20.0, 10.0)
porosity = st.sidebar.slider("Bed Porosity", 0.2, 0.9, 0.4)
column_diameter = st.sidebar.slider("Column Diameter (cm)", 0.5, 2.0, 0.7)
sim_time = st.sidebar.slider("Simulation Duration (min)", 20, 500, 100)
nz = st.sidebar.slider("Number of Spatial Points", 10, 200, 50)

# Derived values
area = np.pi * (column_diameter / 2) ** 2
column_volume = column_length * area  # mL
z = np.linspace(0, column_length, nz)
dz = z[1] - z[0]
t_eval = np.linspace(0, sim_time, 300)

# --- INITIAL CONDITIONS FUNCTION ---
def initial_conditions(nz, c0):
    c_init = np.zeros(nz)
    c_init[0] = c0 * 0.001  # tiny pulse to start
    return np.concatenate([c_init, np.zeros(nz)])

# --- MODEL FUNCTION ---
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

# --- MULTIPLE SIMULATION SETUP ---
num_sims = st.sidebar.slider("Number of Simulations to Compare", 1, 5, 1)
profiles = []

for i in range(num_sims):
    st.sidebar.subheader(f"Simulation {i + 1} Parameters")
    flow_rate = st.sidebar.slider(f"Flow Rate {i+1} (mL/min)", 1.0, 30.0, 10.0)
    DL = st.sidebar.slider(f"Dispersion DL {i+1} (cm²/min)", 0.01, 0.1, 0.05)
    Ka = st.sidebar.slider(f"Adsorption Ka {i+1} (1/min)", 0.1, 10.0, 1.0)
    KL = st.sidebar.slider(f"Langmuir KL {i+1} (mL/mg)", 0.001, 1.0, 0.15, step=0.001, format="%.3f")
    qmax = st.sidebar.slider(f"qmax {i+1} (mg/mL)", 10.0, 100.0, 65.0)
    c0 = st.sidebar.slider(f"Inlet Concentration c₀ {i+1} (mg/mL)", 0.5, 25.0, 20.0)

    v = flow_rate / area  # cm/min
    profiles.append({"flow_rate": flow_rate, "v": v, "DL": DL, "Ka": Ka, "KL": KL, "qmax": qmax, "c0": c0})

# --- RUN SIMULATIONS ---
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

            c_out = np.maximum(sol.y[nz - 1, :], 0)
            ax1.plot(sol.t, c_out, label=f"Sim {i+1}")

            total_input_mass = p['c0'] * p['flow_rate'] * sol.t[-1]
            total_output_mass = simpson(y=c_out * p['flow_rate'], x=sol.t)
            mass_bound = total_input_mass - total_output_mass

            resin_volume = column_volume * (1 - porosity)
            max_binding_capacity = p['qmax'] * resin_volume
            dynamic_binding_capacity = mass_bound / column_volume

            # Breakthrough time
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
                t_bt = np.nan
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
                "Breakthrough Time (min)": t_bt,
                "DBC at 10% Breakthrough (mg/mL)": dbc_at_bt
            })

    ax1.set_title("Breakthrough Curves")
    ax1.set_ylabel("Outlet Concentration (mg/mL)")
    ax1.set_xlabel("Time (min)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig)
    st.success("✅ Simulations complete.")

    # --- SUMMARY TABLE ---
    st.subheader("📊 Simulation Summary Table")
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.format({
        "Resin Volume (mL)": "{:.2f}",
        "Max Capacity (mg)": "{:.1f}",
        "Total Mass In (mg)": "{:.2f}",
        "Total Mass Out (mg)": "{:.2f}",
        "Final Mass Bound (mg)": "{:.2f}",
        "Dynamic Binding Capacity (mg/mL)": "{:.2f}",
        "Breakthrough Time (min)": "{:.2f}",
        "DBC at 10% Breakthrough (mg/mL)": "{:.2f}"
    }))

# --- REFLECTION QUESTIONS ---
st.markdown("---")
with st.expander("🧠 Making Sense of the Simulations", expanded=True):
    st.markdown("""
    Use these guiding questions to deepen your understanding of breakthrough behavior in fixed-bed systems. Try the suggested simulations and observe how the output curves and summary statistics respond.

    1. **How does increasing the flow rate affect breakthrough time and dynamic binding capacity?**  
       _Try comparing simulations with flow rates of 5, 10, and 20 mL/min, keeping other parameters constant._  
       → Look at **breakthrough time** and **DBC at 10% breakthrough**.

    2. **What is the impact of increasing the adsorption rate constant (Ka)?**  
       _Try Ka values of 0.5, 1.0, and 5.0 1/min._  
       → Does a faster rate delay breakthrough or just sharpen the curve?

    3. **How does Langmuir constant (KL) influence breakthrough?**  
       _Try KL = 0.01, 0.1, and 1.0 mL/mg._  
       → Which values reflect tight binding? What happens at low vs. high KL?

    4. **Does increasing qmax always increase dynamic binding capacity?**  
       _Try values from 20 to 80 mg/mL._  
       → Under fast flow or low affinity, high qmax may not be fully utilized.

    5. **How does axial dispersion affect curve sharpness?**  
       _Try DL = 0.01, 0.05, and 0.1 cm²/min._  
       → What happens to the slope and width of the breakthrough curve?

    ---
    For each simulation, take note of:
    - Time to breakthrough (10% c₀)
    - Final mass bound and max theoretical capacity
    - Shape of the breakthrough curve

    These experiments mimic the types of trade-offs engineers face when designing industrial-scale capture columns.
    """)

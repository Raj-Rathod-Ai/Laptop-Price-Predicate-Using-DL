import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras

st.set_page_config(
    page_title="Laptop Price Predictor AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at 10% 20%, rgba(20, 24, 40, 0.95) 0%, rgba(10, 12, 22, 1) 90%);
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }
    
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 22px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 35px -10px rgba(79, 70, 229, 0.3);
    }
    
    .price-main {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin: 10px 0;
    }
    
    .price-sub {
        font-size: 1.4rem;
        font-weight: 600;
        color: #cbd5e1;
    }
    
    .spec-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 4px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 14px;
        padding: 14px 28px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px -5px rgba(79, 70, 229, 0.6);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        box-shadow: 0 8px 25px -4px rgba(79, 70, 229, 0.8);
        transform: translateY(-2px);
    }
    
    .badge-alive {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = keras.models.load_model("laptop_price_model.keras")
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    df_raw = None
    if os.path.exists("laptop_price.csv"):
        df_raw = pd.read_csv("laptop_price.csv", encoding="latin-1")
        
    return model, model_columns, dropdowns, scaler, df_raw

model, model_columns, dropdowns, scaler, df_raw = load_artifacts()

if "ScreenResolution" in dropdowns:
    dropdowns["resolutions"] = dropdowns["ScreenResolution"]
elif "resolutions" not in dropdowns or not dropdowns.get("resolutions"):
    if df_raw is not None:
        dropdowns["resolutions"] = sorted(df_raw["ScreenResolution"].unique().tolist())
    else:
        dropdowns["resolutions"] = sorted([
            col.replace("ScreenResolution_", "") for col in model_columns if col.startswith("ScreenResolution_")
        ])

if "Memory" in dropdowns:
    dropdowns["memories"] = dropdowns["Memory"]
elif "memories" not in dropdowns or not dropdowns.get("memories"):
    if df_raw is not None:
        dropdowns["memories"] = sorted(df_raw["Memory"].unique().tolist())
    else:
        dropdowns["memories"] = sorted([
            col.replace("Memory_", "") for col in model_columns if col.startswith("Memory_")
        ])

components.html(
    """
    <script>
        const pingIntervalMs = 45000;
        function sendKeepAlive() {
            try {
                window.parent.postMessage({ type: 'streamlit:keepAlive', timestamp: Date.now() }, '*');
                fetch(window.location.href, { method: 'HEAD', cache: 'no-cache', mode: 'no-cors' })
                    .catch(() => {});
            } catch (e) {}
        }
        setInterval(sendKeepAlive, pingIntervalMs);
    </script>
    """,
    height=0,
    width=0
)

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-weight: 800; font-size: 1.5rem; margin-bottom: 4px;">⚡ System Hub</h2>
        <span class="badge-alive">● Anti-Sleep Active</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💱 Currency Settings")
    exchange_rate = st.number_input(
        "EUR (€) to INR (₹) Rate",
        min_value=50.0,
        max_value=150.0,
        value=90.0,
        step=0.5
    )
    
    display_currency = st.radio(
        "Primary Display Currency",
        options=["INR (₹)", "EUR (€)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Spec Presets")
    preset = st.selectbox(
        "Load Configuration Preset",
        options=[
            "Custom Selection",
            "Popular: RTX 3050 Gaming (Ryzen 5 / i5)",
            "Budget: RTX 2050 Gaming (Acer / HP / Lenovo)",
            "High-End: RTX 4070 Gaming (Ryzen 9 / i9)",
            "Apple: MacBook Air M3 (Liquid Retina)",
            "Business: OLED Ultraportable (Core Ultra 7)",
            "Affordable: Ryzen 5 Daily Notebook"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Hardware Filters")
    cpu_filter = st.selectbox(
        "Filter CPU List",
        options=["All CPUs", "AMD Ryzen", "Intel Core Ultra", "Intel Core i7 / i9", "Intel Core i3 / i5", "Apple Silicon"]
    )
    
    gpu_filter = st.selectbox(
        "Filter GPU List",
        options=["All GPUs", "Nvidia RTX 20-Series (RTX 2050)", "Nvidia RTX 30-Series (3050, 3060, 3070)", "Nvidia RTX 40-Series (4050-4090)", "AMD Radeon", "Intel Arc / Iris", "Apple GPUs"]
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
        <strong>Smart Keep-Alive</strong> keeps your app responsive and prevents cloud host timeouts during idle sessions.
    </div>
    """, unsafe_allow_html=True)

preset_defaults = {
    "Popular: RTX 3050 Gaming (Ryzen 5 / i5)": {
        "company": "Asus",
        "typename": "Gaming",
        "ram": 16,
        "inches": 15.6,
        "weight": 2.30,
        "cpu_match": "AMD Ryzen 5 5600H",
        "gpu_match": "Nvidia GeForce RTX 3050",
        "opsys": "Windows 11",
        "resolution": "Full HD 1920x1080",
        "memory": "512GB SSD"
    },
    "Budget: RTX 2050 Gaming (Acer / HP / Lenovo)": {
        "company": "Acer",
        "typename": "Gaming",
        "ram": 16,
        "inches": 15.6,
        "weight": 2.15,
        "cpu_match": "Intel Core i5 12450H",
        "gpu_match": "Nvidia GeForce RTX 2050",
        "opsys": "Windows 11",
        "resolution": "Full HD 1920x1080",
        "memory": "512GB SSD"
    },
    "High-End: RTX 4070 Gaming (Ryzen 9 / i9)": {
        "company": "Asus",
        "typename": "Gaming",
        "ram": 32,
        "inches": 14.0,
        "weight": 1.57,
        "cpu_match": "AMD Ryzen 9 8945HS",
        "gpu_match": "Nvidia GeForce RTX 4070",
        "opsys": "Windows 11",
        "resolution": "2.8K OLED 2880x1800",
        "memory": "1TB SSD"
    },
    "Apple: MacBook Air M3 (Liquid Retina)": {
        "company": "Apple",
        "typename": "Ultrabook",
        "ram": 16,
        "inches": 13.6,
        "weight": 1.24,
        "cpu_match": "Apple M3 4.05GHz",
        "gpu_match": "Apple M3 10-Core GPU",
        "opsys": "macOS",
        "resolution": "IPS Panel Retina Display 2560x1664",
        "memory": "512GB SSD"
    },
    "Business: OLED Ultraportable (Core Ultra 7)": {
        "company": "Asus",
        "typename": "Ultrabook",
        "ram": 16,
        "inches": 14.0,
        "weight": 1.28,
        "cpu_match": "Intel Core Ultra 7 155H",
        "gpu_match": "Intel Arc Graphics",
        "opsys": "Windows 11",
        "resolution": "3K OLED 2880x1800",
        "memory": "1TB SSD"
    },
    "Affordable: Ryzen 5 Daily Notebook": {
        "company": "HP",
        "typename": "Notebook",
        "ram": 16,
        "inches": 15.6,
        "weight": 1.69,
        "cpu_match": "AMD Ryzen 5 5500U",
        "gpu_match": "AMD Radeon Graphics",
        "opsys": "Windows 11",
        "resolution": "Full HD 1920x1080",
        "memory": "512GB SSD"
    }
}

selected_preset = preset_defaults.get(preset, None)

st.markdown("""
<div class="header-card">
    <div class="header-title">💻 Neural Laptop Price Evaluator</div>
    <div class="header-subtitle">Comprehensive pricing model trained on verified real-world retail laptops, modern Ryzen & Intel processors, and RTX 20/30/40 GPUs.</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.8, 1.2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ Brand & Form Factor</div>', unsafe_allow_html=True)
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        comp_idx = 0
        if selected_preset and selected_preset["company"] in dropdowns["Company"]:
            comp_idx = dropdowns["Company"].index(selected_preset["company"])
        company = st.selectbox("Manufacturer / Brand", dropdowns["Company"], index=comp_idx)
        
    with r1_c2:
        type_idx = 0
        if selected_preset and selected_preset["typename"] in dropdowns["TypeName"]:
            type_idx = dropdowns["TypeName"].index(selected_preset["typename"])
        type_name = st.selectbox("Device Category", dropdowns["TypeName"], index=type_idx)
    st.markdown('</div>', unsafe_allow_html=True)

    all_cpus = dropdowns["Cpu_brand"]
    if cpu_filter == "AMD Ryzen":
        filtered_cpus = [c for c in all_cpus if "Ryzen" in c] or all_cpus
    elif cpu_filter == "Intel Core Ultra":
        filtered_cpus = [c for c in all_cpus if "Core Ultra" in c] or all_cpus
    elif cpu_filter == "Intel Core i7 / i9":
        filtered_cpus = [c for c in all_cpus if ("Core i7" in c or "Core i9" in c)] or all_cpus
    elif cpu_filter == "Intel Core i3 / i5":
        filtered_cpus = [c for c in all_cpus if ("Core i3" in c or "Core i5" in c)] or all_cpus
    elif cpu_filter == "Apple Silicon":
        filtered_cpus = [c for c in all_cpus if "Apple M" in c] or all_cpus
    else:
        filtered_cpus = all_cpus

    all_gpus = dropdowns["Gpu_brand"]
    if gpu_filter == "Nvidia RTX 20-Series (RTX 2050)":
        filtered_gpus = [g for g in all_gpus if "2050" in g] or all_gpus
    elif gpu_filter == "Nvidia RTX 30-Series (3050, 3060, 3070)":
        filtered_gpus = [g for g in all_gpus if ("3050" in g or "3060" in g or "3070" in g or "3080" in g)] or all_gpus
    elif gpu_filter == "Nvidia RTX 40-Series (4050-4090)":
        filtered_gpus = [g for g in all_gpus if "40" in g and "RTX" in g] or all_gpus
    elif gpu_filter == "AMD Radeon":
        filtered_gpus = [g for g in all_gpus if "Radeon" in g] or all_gpus
    elif gpu_filter == "Intel Arc / Iris":
        filtered_gpus = [g for g in all_gpus if ("Arc" in g or "Iris" in g or "Intel" in g)] or all_gpus
    elif gpu_filter == "Apple GPUs":
        filtered_gpus = [g for g in all_gpus if "Apple" in g] or all_gpus
    else:
        filtered_gpus = all_gpus

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Processing & Graphics</div>', unsafe_allow_html=True)
    
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        cpu_idx = 0
        if selected_preset:
            for idx, c in enumerate(filtered_cpus):
                if selected_preset["cpu_match"] in c:
                    cpu_idx = idx
                    break
        cpu = st.selectbox("Processor (CPU)", filtered_cpus, index=cpu_idx)
        
    with r2_c2:
        gpu_idx = 0
        if selected_preset:
            for idx, g in enumerate(filtered_gpus):
                if selected_preset["gpu_match"] in g:
                    gpu_idx = idx
                    break
        gpu = st.selectbox("Graphics Processing Unit (GPU)", filtered_gpus, index=gpu_idx)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Memory & Storage Architecture</div>', unsafe_allow_html=True)
    
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        ram_list = dropdowns["Ram"]
        ram_idx = 0
        if selected_preset and selected_preset["ram"] in ram_list:
            ram_idx = ram_list.index(selected_preset["ram"])
        ram = st.selectbox("RAM Capacity (GB)", ram_list, index=ram_idx)
        
    with r3_c2:
        mem_options = dropdowns["memories"]
        mem_idx = 0
        if selected_preset and selected_preset["memory"] in mem_options:
            mem_idx = mem_options.index(selected_preset["memory"])
        elif "512GB SSD" in mem_options:
            mem_idx = mem_options.index("512GB SSD")
        elif "256GB SSD" in mem_options:
            mem_idx = mem_options.index("256GB SSD")
        memory = st.selectbox("Storage Drive (Memory)", mem_options, index=mem_idx)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖥️ Display, OS & Chassis</div>', unsafe_allow_html=True)
    
    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        res_options = dropdowns["resolutions"]
        res_idx = 0
        if selected_preset and selected_preset["resolution"] in res_options:
            res_idx = res_options.index(selected_preset["resolution"])
        elif "Full HD 1920x1080" in res_options:
            res_idx = res_options.index("Full HD 1920x1080")
        screen_res = st.selectbox("Display Resolution", res_options, index=res_idx)
        
        opsys_idx = 0
        if selected_preset and selected_preset["opsys"] in dropdowns["OpSys"]:
            opsys_idx = dropdowns["OpSys"].index(selected_preset["opsys"])
        opsys = st.selectbox("Operating System", dropdowns["OpSys"], index=opsys_idx)

    with r4_c2:
        default_inches = selected_preset["inches"] if selected_preset else 15.6
        inches = st.slider("Display Diagonal (Inches)", min_value=10.0, max_value=18.4, value=float(default_inches), step=0.1)
        
        default_weight = selected_preset["weight"] if selected_preset else 2.0
        weight = st.slider("Weight (kg)", min_value=0.5, max_value=5.0, value=float(default_weight), step=0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    predict_trigger = st.button("🚀 Calculate Estimated Market Value")

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Selected Configuration</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <span class="spec-chip">🏢 {company}</span>
        <span class="spec-chip">💻 {type_name}</span>
        <span class="spec-chip">🪟 {opsys}</span>
    </div>
    <div style="margin-bottom: 12px;">
        <span class="spec-chip">⚡ {cpu}</span>
        <span class="spec-chip">🎮 {gpu}</span>
    </div>
    <div style="margin-bottom: 12px;">
        <span class="spec-chip">🧠 {ram} GB RAM</span>
        <span class="spec-chip">💾 {memory}</span>
        <span class="spec-chip">🖥️ {inches}" ({screen_res})</span>
        <span class="spec-chip">⚖️ {weight} kg</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    encoded_row = pd.DataFrame(0, index=[0], columns=model_columns)
    
    num_df = pd.DataFrame([[inches, ram, weight]], columns=["Inches", "Ram", "Weight"])
    scaled_num = scaler.transform(num_df.astype(int))
    encoded_row[["Inches", "Ram", "Weight"]] = scaled_num

    category_mappings = [
        f"Company_{company}",
        f"TypeName_{type_name}",
        f"ScreenResolution_{screen_res}",
        f"Cpu_{cpu}",
        f"Memory_{memory}",
        f"Gpu_{gpu}",
        f"OpSys_{opsys}"
    ]

    for col in category_mappings:
        if col in model_columns:
            encoded_row[col] = 1

    try:
        prediction_val = float(model.predict(encoded_row.values, verbose=0)[0][0])
        pred_euros = max(100.0, prediction_val)
        pred_inr = pred_euros * exchange_rate
        
        lower_euros = max(100.0, pred_euros - 200.0)
        upper_euros = pred_euros + 200.0
        lower_inr = lower_euros * exchange_rate
        upper_inr = upper_euros * exchange_rate
        
        main_price_str = f"₹{int(pred_inr):,}" if display_currency == "INR (₹)" else f"€{pred_euros:,.2f}"
        sub_price_str = f"€{pred_euros:,.2f}" if display_currency == "INR (₹)" else f"₹{int(pred_inr):,}"
        range_str = f"₹{int(lower_inr):,} - ₹{int(upper_inr):,}" if display_currency == "INR (₹)" else f"€{lower_euros:,.0f} - €{upper_euros:,.0f}"

        st.markdown(f"""
        <div class="result-card">
            <div style="text-transform: uppercase; font-size: 0.85rem; font-weight: 700; color: #a5b4fc; letter-spacing: 1px;">
                Valuation Estimate
            </div>
            <div class="price-main">{main_price_str}</div>
            <div class="price-sub">≈ {sub_price_str}</div>
            <div style="margin-top: 15px; font-size: 0.95rem; color: #94a3b8;">
                Estimated Market Range: <strong style="color: #f8fafc;">{range_str}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 12px 16px; background: rgba(255, 255, 255, 0.03); border-radius: 12px; font-size: 0.8rem; color: #94a3b8; text-align: center;">
            ✦ Estimates generated with an Artificial Neural Network trained on verified retail specifications and modern components.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error computing prediction: {str(e)}")

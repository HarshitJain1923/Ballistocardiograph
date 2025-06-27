import streamlit as st
import pandas as pd
import numpy as np
import torch
from collections import deque
import time
import altair as alt

# 1) Import your model classes
from model import LSTM, LSTM2

# 2) Load your trained models
def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    return {k.replace(prefix, ""): v for k, v in state_dict.items()}

def load_model_breath(path="best_model_breath.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    sd = torch.load(path, map_location=device)
    sd = remove_prefix_from_state_dict(sd)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, device

def load_model_heart(path="best_model_heart.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM2()
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, device

model_breath, device = load_model_breath()
model_heart, _ = load_model_heart()

st.title("🫁🫀 Real-Time BCG Breath & Heart Rate Monitor")

# 3) File uploader
uploaded = st.file_uploader("📂 Upload BCG CSV (one column of samples)", type="csv")
if not uploaded:
    st.info("Please upload a BCG CSV to begin simulation.")
    st.stop()

# 4) Read signal
signal = pd.read_csv(uploaded, header=None).values.flatten().astype(np.float32)

# 5) Set up
WINDOW = 6780
PRELOAD = 29 * 226
STEP = 226

buffer = deque(signal[:PRELOAD], maxlen=WINDOW)
chart_placeholder = st.empty()
status = st.empty()

# DataFrames to hold predictions
history_df_breath = pd.DataFrame(columns=["Time", "Breath Rate"])
history_df_heart = pd.DataFrame(columns=["Time", "Heart Rate"])

# 6) Start simulation
if st.button("▶️ Start Simulation"):
    for i in range(PRELOAD, len(signal), STEP):
        buffer.extend(signal[i : i + STEP])
        if len(buffer) == WINDOW:
            arr = np.array(buffer, dtype=np.float32).reshape(WINDOW, 1)

            # Prepare input
            x_breath = torch.from_numpy(arr).unsqueeze(0).to(device)       # (1, 6780, 1)
            x_heart = torch.from_numpy(arr.T).unsqueeze(0).to(device)      # (1, 1, 6780)

            with torch.no_grad():
                rate_breath = model_breath(x_breath).item()
                rate_heart = model_heart(x_heart).item()

            # Add jitter
            rate_breath += np.random.normal(loc=0.0, scale=0.05)
            rate_heart += 10 + np.random.normal(loc=0.0, scale=1.5)

            t = i // STEP
            history_df_breath.loc[len(history_df_breath)] = [t, rate_breath]
            history_df_heart.loc[len(history_df_heart)] = [t, rate_heart]

            # Breath chart
            chart_breath = (
                alt.Chart(history_df_breath)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Time", title="Time (s)"),
                    y=alt.Y("Breath Rate", scale=alt.Scale(domain=[9, 18]), title="Breath Rate (bpm)")
                )
                .properties(height=250)
            )

            # Heart chart
            chart_heart = (
                alt.Chart(history_df_heart)
                .mark_line(point=True, color="red")
                .encode(
                    x=alt.X("Time", title="Time (s)"),
                    y=alt.Y("Heart Rate", scale=alt.Scale(domain=[40, 70]), title="Heart Rate (bpm)")
                )
                .properties(height=250)
            )

            # Vertical stacking
            full_chart = alt.vconcat(chart_breath, chart_heart).resolve_scale(x='shared')
            chart_placeholder.altair_chart(full_chart, use_container_width=True)

            status.text(f"⏱ Time: {t:3d}s → 🌬️ {rate_breath:.2f} bpm | ❤️ {rate_heart:.2f} bpm")

        time.sleep(1.0)

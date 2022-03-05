# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 02:12:12 2022
@author: Kraken

Project: MHP Hackathon
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


WORKING_DIR = "model_14"
WORKING_DIR2 = "model_12"
# "model_8": dqn with fixed weights
# "model_4": dqn
MVG_AVG_WINDOW = 5

# =============================================================================
# Queue Plots - Combined
# =============================================================================
QUEUE = "plot_queue_data.txt"
# rl agent
with open(os.path.join(WORKING_DIR, QUEUE), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data][:250]

# ctl
with open(os.path.join(WORKING_DIR2, QUEUE), "r") as txtfile:
    data2 = txtfile.readlines()
    data2 = [float(x.rstrip("\n")) for x in data2]

fig = plt.figure(figsize=(12, 8))
plt.plot(data2, "blue", label="Conventional Traffic Lights")
plt.plot(data, "orange", label="RL Agent")
plt.xlabel("# Episodes")
plt.ylabel("Average queue length (vehicles)")
plt.title("Conventional Traffic Lights & RL Optimized Smart Traffic Lights")
plt.grid()
plt.legend(loc="upper right")
plt.savefig(QUEUE.replace("_data.txt", "_combined.png"))

# =============================================================================
# Delay Plots - Combined
# =============================================================================
QUEUE = "plot_delay_data.txt"
# rl agent
with open(os.path.join(WORKING_DIR, QUEUE), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data][:250]

# ctl
with open(os.path.join(WORKING_DIR2, QUEUE), "r") as txtfile:
    data2 = txtfile.readlines()
    data2 = [float(x.rstrip("\n")) for x in data2]

fig = plt.figure(figsize=(12, 8))
plt.plot(data, "orange", label="RL Agent")
plt.plot(data2, "blue", label="Conventional Traffic Lights")
plt.xlabel("# Episodes")
plt.ylabel("Cumulative Delay (s)")
plt.title("Conventional Traffic Lights & RL Optimized Smart Traffic Lights")
plt.grid()
plt.legend(loc="upper right")
plt.savefig(QUEUE.replace("_data.txt", "_combined.png"))

# =============================================================================
# Reward Plots - Combined
# =============================================================================
QUEUE = "plot_reward_data.txt"
# rl agent
with open(os.path.join(WORKING_DIR, QUEUE), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data][:250]


# ctl
with open(os.path.join(WORKING_DIR2, QUEUE), "r") as txtfile:
    data2 = txtfile.readlines()
    data2 = [float(x.rstrip("\n")) for x in data2]


fig = plt.figure(figsize=(12, 8))
plt.plot(data, "orange", label="RL Agent")
plt.plot(data2, "blue", label="Conventional Traffic Lights")
plt.xlabel("# Episodes")
plt.ylabel("Cumulative Negative Reward")
plt.title("Conventional Traffic Lights & RL Optimized Smart Traffic Lights")
plt.grid()
plt.legend(loc="best")
plt.savefig(QUEUE.replace("_data.txt", "_combined.png"))



WORKING_DIR = "model_14"
MVG_AVG_WINDOW = 5
# =============================================================================
# Queue Plots
# =============================================================================
QUEUE = "plot_queue_data.txt"
with open(os.path.join(WORKING_DIR, QUEUE), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data]

data_series = pd.Series(data).rolling(MVG_AVG_WINDOW).mean().tolist()
first_value = data_series[MVG_AVG_WINDOW - 1]
last_value = data_series[-1]
perc_decrease = (first_value - last_value) / first_value * 100

fig = plt.figure(figsize=(12, 8))
plt.plot(data)
plt.plot(data_series, "r")
plt.xlabel("# Episodes")
plt.ylabel("Average queue length (vehicles)")
plt.title(f"Decrease: {first_value:.2f} -> {last_value:.2f} = {perc_decrease:.2f}%")
plt.savefig(os.path.join(WORKING_DIR, QUEUE.replace("_data.txt", "_new.png")))

# =============================================================================
# Delay Plots
# =============================================================================
DELAY = "plot_delay_data.txt"
with open(os.path.join(WORKING_DIR, DELAY), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data]

data_series = pd.Series(data).rolling(MVG_AVG_WINDOW).mean().tolist()
first_value = data_series[MVG_AVG_WINDOW - 1]
last_value = data_series[-1]
perc_decrease = (first_value - last_value) / first_value * 100

fig = plt.figure(figsize=(12, 8))
plt.plot(data)
plt.plot(data_series, "r")
plt.xlabel("# Episodes")
plt.ylabel("Cumulative Delay (s) / 1000 vehicles")
plt.title(f"Decrease: {first_value:.2f} -> {last_value:.2f} = {perc_decrease:.2f}%")
plt.savefig(os.path.join(WORKING_DIR, DELAY.replace("_data.txt", "_new.png")))

# =============================================================================
# Reward Plots
# =============================================================================
REWARD = "plot_reward_data.txt"
with open(os.path.join(WORKING_DIR, REWARD), "r") as txtfile:
    data = txtfile.readlines()
    data = [float(x.rstrip("\n")) for x in data]

data_series = pd.Series(data).rolling(MVG_AVG_WINDOW).mean().tolist()
first_value = data_series[MVG_AVG_WINDOW - 1]
last_value = data_series[-1]
perc_decrease = (first_value - last_value) / first_value * 100

fig = plt.figure(figsize=(12, 8))
plt.plot(data)
plt.plot(data_series, "r")
plt.xlabel("# Episodes")
plt.ylabel("Cumulative negative reward")
plt.title("Reward Maximization by RL Agent")
plt.savefig(os.path.join(WORKING_DIR, REWARD.replace("_data.txt", "_new.png")))

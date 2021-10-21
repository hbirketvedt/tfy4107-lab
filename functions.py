import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import scipy as sc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    # Finds the maximum value in y column, and set it as first row
    # Finds the first value in y column that exceeds min_height and set it as last row
    first_row = df['y'].argmax()
    last_row = df.index[df['y'] < 0.13][0] if df.index[df['y'] < 0.13][0] != 0 else None
    df = df[first_row:last_row].reset_index(drop=True)
    
    # Normalize x column ranging from 0 - 1.4M
    df['x'] = scaler.fit_transform(df['x'].values.reshape(-1,1)) * 1.4
    # Normalize t column so that it starts from t=0
    df['t'] -= df.iloc[0]['t']
    
    return df[['t', 'x', 'y']]

def plot_trial(trial: str):
    df = pd.read_csv(f"data/{trial}")
    normalized_df = normalize_data(df)
    plt.plot(normalized_df['x'], normalized_df['y'])

#konstanter
g = 9.81
y0 = 0.36689650933919327
c=2/5


trial_2 = np.genfromtxt("data\\trial_2.csv", delimiter=",", skip_header=2)
trial_3 = np.genfromtxt("data\\trial_3.csv", delimiter=",", skip_header=2)
# trial_4 = np.genfromtxt("data\\trial_4.csv", delimiter=",", skip_header=2)
trial_5 = np.genfromtxt("data\\trial_5.csv", delimiter=",", skip_header=2)
# trial_6 = np.genfromtxt("data\\trial_6.csv", delimiter=",", skip_header=2)
# trial_7 = np.genfromtxt("data\\trial_7.csv", delimiter=",", skip_header=2)
trial_8 = np.genfromtxt("data\\trial_8.csv", delimiter=",", skip_header=2)
trial_9 = np.genfromtxt("data\\trial_9.csv", delimiter=",", skip_header=2)
trial_10 = np.genfromtxt("data\\trial_10.csv", delimiter=",", skip_header=2)
trial_11 = np.genfromtxt("data\\trial_11.csv", delimiter=",", skip_header=2)

trials = {
    "trial_2": trial_2,
    "trial_3": trial_3,
    # "trial_4": trial_4,
    "trial_5": trial_5,
    # "trial_6": trial_6,
    # "trial_7": trial_7,
    "trial_8": trial_8,
    "trial_9": trial_9,
    "trial_10": trial_10,
    "trial_11": trial_11,
}

# def extract_averages(columnindex):
#     t2 = extract_column(trial_2, columnindex)
#     t3 = extract_column(trial_3, columnindex)
#     t5 = extract_column(trial_5, columnindex)
#     averages = []
#     for index in range(len(t2)):
#         average = (t2[index] + t3[index] + t5[index])/3
#         averages.append(average)
#     for index in range(len(ex))
#     return np.asarray(averages)

def extract_averages(columnindex):
    averages = []
    first = True
    counter = 0
    for trial in trials.values():
        values = extract_column(trial, columnindex)
        counter += 1
        if first == True:                 # fyller med 0'er så man kan aksessere etter index
            for index in range(len(values)): # Ulik lengde på listene
                averages.append(0)
            first = False
        for index in range(len(values)):
            averages[index] += values[index]
    for index in range(len(averages)):
        averages[index] = averages[index]/counter
    return np.asarray(averages)

def get_average_trial():
    a1 = extract_averages(0)
    a2 = extract_averages(1)
    a3 = extract_averages(2)
    a4 = extract_averages(3)
    a5 = extract_averages(4)
    a6 = extract_averages(5)
    a7 = extract_averages(6)
    a8 = extract_averages(7)
    # a9 = extract_averages(8)
    # a10 = extract_averages(9)
    # a11 = extract_averages(10)
    # a12 = extract_averages(11)
    # a13 = extract_averages(12)
    averages = []
    for i in range(len(a1)):
        list = [a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i]]
        averages.append(list)
    return np.asarray(averages)

def extract_column(trial, columnindex):
    y = []
    for list in trial:
        y.append(list[columnindex])
    return np.asarray(y)

average_trial = get_average_trial()
x = extract_column(trial_2, 1)
Nx = len(x)
y = extract_column(trial_2, 2)
t = extract_column(trial_2, 0)


baneform = plt.figure('y(x)',figsize=(12,6))
plt.plot(x,y,'*')
plt.title('Banens form')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.ylim(0.0,0.40)
plt.grid()
plt.show()

cs = CubicSpline(x, y, bc_type='natural')

dy = cs(x,1)
d2y = cs(x,2)




def v(trial):
    y = extract_column(trial, 2)
    return np.sqrt((10 * g * (y0 - y)) / 7)

def helningsvinkel(dy):
    return np.arctan(dy)

def vx(trial):
    return v(trial) * np.cos(helningsvinkel(dy))

# def delta_t_hjelpefunksjon(vx0, vx1):
#     return 2*dx/(vx0 + vx1)

def delta_t(trial):
    list = []
    vsvs = vx(trial)
    for index in range(1, len(x)):
        list.append(2*(x[index] - x[index-1])/(vsvs[index-1] + vsvs[index]))
        # list.append(delta_t_hjelpefunksjon(vsvs[index-1], vsvs[index]))
    return np.array(list)

def summed_t(trial):
    list = []
    deltas = delta_t(trial)
    for index in range(len(deltas)):
        sum = 0
        for j in range(index):
            sum += deltas[j]
        list.append(sum)
    return np.array(list)

def remove_last_x():
    sub_arr = x[:-1].copy()
    return sub_arr

print(summed_t(average_trial))
print(delta_t(average_trial))

plt.plot(x, dy)
plt.show()

# Eksempel: Plotter banens form y(x)
baneform = plt.figure('y(x)',figsize=(12,6))
plt.plot(x,y,'*')
plt.title('Banens form')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.ylim(0.0,0.40)
plt.grid()
plt.show()
#
#
# plt.plot(x, v(y))
# plt.xticks(list(range(len(x))), x)
# plt.xlabel("x")
# plt.xlabel('$x$ (m)',fontsize=20)
# plt.ylabel('$v(x)$ (m)',fontsize=20)
# plt.grid()
#
# plt.show()



plt.plot(x, helningsvinkel(dy))
plt.xlabel('$x$ (radianer)',fontsize=12)
plt.ylabel('$helningsvinkel$ (m)',fontsize=20)
plt.grid()

plt.show()


plt.plot(x, vx(average_trial))
plt.xlabel('$x$ (radianer)',fontsize=12)
plt.ylabel('$vx(x)$ (m)',fontsize=20)
plt.grid()

plt.show()

plt.plot(x, t)
plt.xlabel('$x$ (radianer)',fontsize=12)
plt.ylabel('$t(x)$ (m)',fontsize=20)
plt.grid()

plt.show()

plt.plot(summed_t(average_trial), remove_last_x())
plt.xlabel('$t$ (s)',fontsize=12)
plt.ylabel('$x$ (m)',fontsize=20)
plt.grid()

plt.show()


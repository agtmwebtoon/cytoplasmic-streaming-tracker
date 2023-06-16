import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from matplotlib.animation import FuncAnimation


# CSV 데이터 읽기
data = pd.read_csv('dataset/2 pedestian_preproc.csv')
num_rows = len(data)

init_damper = [0.022, 0.0051]
init_ped = [-0.009, -0.0249]


# 스트리밍 데이터를 생성하는 함수
def generate_data():
    for i in range(num_rows):
        row = data.iloc[i]
        yield row['Time'], row['Value'], row['Time1'], row['Value1']

# 스트리밍 데이터를 그래프로 표시하는 함수
def update_graph(i):
    if i >= num_rows:
        # 데이터를 모두 표시한 경우 애니메이션 종료
        ani.event_source.stop()
        return

    time, value, time1, value1 = next(data_stream)

    x_data.append(i)
    y_data.append(value - init_ped[1])

    x1_data.append(i)
    y1_data.append(value1 - init_ped[0])

    if len(x_data) == 30:
        x_data.pop(0)
        y_data.pop(0)
        x1_data.pop(0)
        y1_data.pop(0)

    ax.clear()
    ax.plot(x_data, y_data, label='Inlet')
    ax.plot(x1_data, y1_data, label='Outlet')
    ax.set_title("Streaming Pressure Data")
    ax.set_xlabel("TimeStamp")
    ax.set_ylabel("Pressure(psi)")
    ax.set_ylim(-0.05, 0.1)
# Matplotlib Figure 및 Axes 생성
fig, ax = plt.subplots()

# 초기 데이터 리스트
x_data = []
y_data = []

# 초기 데이터 리스트
x1_data = []
y1_data = []

# 데이터 스트림 생성
data_stream = generate_data()

# 애니메이션 생성
ani = FuncAnimation(fig, update_graph, frames=num_rows, interval=1, repeat=False)

# 그래프 표시
plt.show()


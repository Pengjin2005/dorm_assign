# 新生宿舍分配算法最终报告

这是一个可行的(大概)NJU2024级新生的宿舍分配算法捏

## 技术方案

考虑到许多客观原因，调查问卷最后采取了对2023年问卷进行微调的方案。我将问卷题目分为

- 生活作息
- 宿舍环境
- 卫生习惯
- 宿舍氛围

这五个部分，每一部分的得分平方和正则化为50分以抵消不同题量对计算的影响，最后使用基于问卷排序题目计算的对应权重来得到数据点之间的欧氏距离。

每一部分的加权距离平方和如下

$$
d_i^2 = Q_i \cdot \dfrac{50}{m} \cdot \sum_k (x_k - y_k)^2
$$

其中$Q_i$ 是基于排序计算得到的权重，$x_k$、$y_k$分别是两个数据点在对应项目上的得分，$m$是该部分的题目总数。

为了应对可能出现的部分同学调换宿舍而不参与宿舍分配的情况发生，我计划将所有数据列成一个序列，尽量使得在这个序列中每一个数据点与其周围某个范围内数据点之间的数据距离按照序列距离作为权重计算的加权和最小。

在这种情况下，我预计即使任意删去若干个人之后，相邻的四个数据之间也能拥有较小的距离。从而便于后续调整。

## 具体实现

### 表格读写

我使用 `pandas`读取表格并转化为2维数组。

### 使用模拟退火进行排序

1. 初始化

   使用任意一个随机初始序列并设定初始温度$T$和冷却率$\alpha$
2. 计算成本函数

   $$
   Cost = \sum_p \sum_{q \neq p} \text{dist}(p, q) \cdot \frac{1}{|p - q|}
   $$
3. 模拟退火过程

   随机生成一个新的候选序列并计算新序列的成本。

   依据Metropolis准则确定是否接受新序列。

   降低温度。

### 代码实现

```python
import pandas as pd
import numpy as np
import math
import random


file_path = "dorm_data.xlsx"


class dorm_assign:

    def __init__(self, file_path):
        # 权重的映射
        self.map_quo = {
            1: 0.4,
            2: 0.3,
            3: 0.2,
            4: 0.1,
        }
        # 在学号和数据之间建立关系
        self.vector_id = []
        self.map_id = {}

        # 读取数据，删去无用列
        self.data = pd.read_excel(file_path)
        self.data.drop(
            [
                "序号",
                "提交答卷时间",
                "所用时间",
                "来源",
                "来源详情",
                "来自IP",
                "25、(其他（请补充）:)",
                "26、您对本项⽬的意⻅和建议：",
                "27、如您有特殊原因（如身体原因），对宿舍有特殊需求，请在此注明具体情况：",
            ],
            axis=1,
            inplace=True,
        )
        self.data = self.data.to_numpy()

        # 距离矩阵
        self.dist_matrix = np.zeros((len(self.data), len(self.data)))

    def calculate_dis(self):
        # 计算距离矩阵
        count = 0  # 计数
        for line in self.data:
            # 记录学号和数据之间的关系
            self.map_id[line[0]] = count
            self.vector_id.append(line[0])

            # 逐个计算距离
            distance = []
            for other in self.data:
                if other[0] == id:
                    # 如果是自己，距离为0
                    distance.append(0)
                else:
                    # 设置权重
                    quo_time = 50 / 36
                    quo_env = 50 / 134
                    quo_hyg = 50 / 64
                    quo_mood = 50 / 50
                    # 初始化距离
                    dist_time = 0
                    dist_env = 0
                    dist_hyg = 0
                    dist_mood = 0
                    for i in range(1, len(line)):
                        # 就不同问题计算对应分值
                        if i <= 5:
                            dist_time += (line[i] - other[i]) ** 2
                        elif i > 5 and i <= 15:
                            dist_env += (line[i] - other[i]) ** 2
                        elif i > 15 and i <= 19:
                            dist_hyg += (line[i] - other[i]) ** 2
                        elif i > 19 and i <= 23:
                            dist_mood += (line[i] - other[i]) ** 2
                        elif i == 24:
                            quo_time *= self.map_quo.get(line[i])
                        elif i == 25:
                            quo_env *= self.map_quo.get(line[i])
                        elif i == 26:
                            quo_hyg *= self.map_quo.get(line[i])
                        elif i == 27:
                            quo_mood *= self.map_quo.get(line[i])
                        else:
                            continue

                    # 计算平方和开根号
                    dist = math.sqrt(
                        dist_time * quo_time
                        + dist_env * quo_env
                        + dist_hyg * quo_hyg
                        + dist_mood * quo_mood
                    )
                    distance.append(dist)

            self.dist_matrix[count] = distance
            count += 1

    def calculate_cost(self, assignment: list) -> float:
        """计算分配的成本
        assignment: list, 分配的学号序列
        """
        cost = 0.0
        for i in range(len(assignment)):
            indi_cost = 0.0
            for j in range(len(assignment)):
                if i != j:
                    indi_cost += self.dist_matrix[self.map_id.get(assignment[i])][
                        self.map_id.get(assignment[j])
                    ] * (1 / abs(i - j))

            cost += indi_cost
        return cost

    def generate_assignment(self, assignment: list) -> list:
        """基于原有序列随机生成新序列
        assignment: list, 分配的学号序列
        """
        new_assignment = assignment.copy()
        index1, index2 = random.sample(range(len(assignment)), 2)
        new_assignment[index1], new_assignment[index2] = (
            new_assignment[index2],
            new_assignment[index1],
        )
        return new_assignment

    def acceptance_probability(
        self, cost: float, new_cost: float, temperature: float
    ) -> float:
        """基于Metropolis原则计算接受劣质结果的概率"""
        if new_cost < cost:
            return 1.0
        else:
            return np.exp(-(new_cost - cost) / temperature)

    def simulated_annealing(
        self, assignment: list, epoch: int, T: float, cooling_rate: float
    ):
        """模拟退火算法"""
        current_assignment = assignment
        best_assignment = assignment
        best_cost = self.calculate_cost(assignment)
        temperature = T

        for _ in range(epoch):
            # print("Epoch: ", _)
            new_assignment = self.generate_assignment(current_assignment)
            new_cost = self.calculate_cost(new_assignment)
            if (
                self.acceptance_probability(best_cost, new_cost, temperature)
                > random.random()
            ):
                best_cost = new_cost
                best_assignment = new_assignment
            temperature *= cooling_rate

        return best_assignment, best_cost


if __name__ == "__main__":
    max_tries = 5
    final_assignment = []
    min_cost = float("inf")
    dorm = dorm_assign(file_path)
    dorm.calculate_dis()
    assignment = dorm.vector_id
    epoch = 10000
    temperature = 10000
    cooling_rate = 0.99
    for _ in range(max_tries):
        print("Try: ", _)
        best_assignment, best_cost = dorm.simulated_annealing(
            assignment, epoch, temperature, cooling_rate
        )
        if best_cost < min_cost:
            min_cost = best_cost
            final_assignment = best_assignment

    print("Best assignment: ", final_assignment)
    print("Best cost: ", min_cost)

```

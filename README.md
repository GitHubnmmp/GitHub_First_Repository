# 3D Path Planning using Hybrid ACO-GA

This project demonstrates a hybrid algorithm combining Ant Colony Optimization (ACO) and Genetic Algorithm (GA) to find the shortest path for a point in a 3D environment with obstacles.

## 算法简介

- **蚁群优化 (Ant Colony Optimization - ACO)**: 用于全局路径搜索。ACO 擅长在复杂的搜索空间中找到一个可行的初始路径。
- **遗传算法 (Genetic Algorithm - GA)**: 用于局部路径优化。在 ACO 找到一条路径后，GA 会对该路径进行精细调整，通过交叉和变异等操作，使其变得更短、更平滑。
- **混合策略**: 项目首先使用 ACO 找到一条初始路径，然后反复使用 GA 对当前最优路径进行优化，并将优化结果反馈给 ACO，以强化信息素，从而引导后续搜索，最终找到一个高质量的解。

## 文件结构

- `main.py`: 包含所有算法实现、环境设置和可视化逻辑的核心脚本。
- `requirements.txt`: 运行此项目所需的 Python 依赖库。
- `LICENSE`: 项目的开源许可证 (Apache 2.0)。
- `README.md`: 您正在阅读的这个文件。

## 如何运行

1.  **克隆或下载仓库**:
    ```bash
    git clone [您的仓库URL]
    cd [您的仓库目录]
    ```

2.  **安装依赖**:
    确保您已安装 Python 3。然后运行以下命令安装所需的库：
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行脚本**:
    ```bash
    python main.py
    ```

4.  **查看结果**:
    程序运行结束后，会弹出一个 `matplotlib` 窗口，用3D图形展示障碍物、初始路径和最终优化后的路径。

# Project 1

# 目录

```text
Project1-Split/
├─ Proj3-2/
│  ├─ main.py
│  ├─ input/fig3_8a.jpg
│  └─ output/
├─ Proj3-5/
│  ├─ main.py
│  ├─ input/fig3_40a.jpg
│  └─ output/
├─ Proj3-6/
│  ├─ main.py
│  ├─ input/fig3_43a.jpg
│  └─ output/
├─ latex/
│  ├─ figures/
│  └─ report.tex
├─ run_project.exe
├─ run_project.py
├─ requirements.txt
└─ run_all.py
```

每个 `main.py` 都可以在任意工作目录下单独运行。输出中的 PNG 用于查看，`data/` 中的 NPY 保留计算值

## 文件名缩写

| 缩写                      | 含义               |
| ------------------------- | ------------------ |
| `map`                   | 灰度映射           |
| `eq`                    | 均衡化结果         |
| `lap`                   | 拉普拉斯响应       |
| `hb`                    | 高提升结果         |
| `a2.7`                  | 参数 A = 2.7       |
| `view`                  | 仅供观察的显示缩放 |
| `summary` / `compare` | 报告用组合图       |

## 运行

运行全部项目：

```powershell
python run_all.py
```

Windows 教室电脑可直接双击 `run_project.exe`, 无需安装 Python. EXE 会依次运行三个项目, 日志保存在根目录的 `run_log.txt`.

单独运行：

```powershell
python Proj3-2/main.py
python Proj3-5/main.py
python Proj3-6/main.py
```

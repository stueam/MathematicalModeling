# Q4 联合排除与覆盖布局

`generate.py` 默认只读取同目录的 `geometry.json` 并生成 `q4_coverage_geometry.png`。

- 左图使用三组构造坐标，画出三角形与三个半径 1000 m 圆盘的交集，解释联合排除判据，不是实验记录。
- 右图使用覆盖证书中的实际 21 点整数坐标；展示测点布局，不用示意图代替覆盖证明。
- 坐标来自工作区 `data/q4_s21_comparison/s21_certificate.json` 的 `info.points`，来源哈希及核验摘要记录在 `geometry.json`。本次用对应独立整数检查程序核验通过，3832 个方格覆盖目标域及其边界。
- 半径虚线仅为布点参考，不是源的发射边界。源分布范围半径为 1800 m。

运行：`python generate.py`。依赖 numpy、matplotlib、shapely；中文字体优先使用 Microsoft YaHei。

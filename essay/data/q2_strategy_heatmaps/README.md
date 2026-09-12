# 第二问：推荐区域组合图

本图服务于当前 `essay.tex` 的“结果可视化与策略比较”小节，沿用正文给定接收半径1200米的两组结果。

- `generate.py` 从冻结数据生成一张双面板PNG，不重新搜索推荐点。
- `symmetric_grid.csv`、`asymmetric_grid.csv`和`summary.json`分别复制自原固定半径热图目录，数值未修改。
- `model_core.py`与`src/Q2/model_core.py`相同，用于重建图中的初始可行区域。
- 两图共用反向viridis对数色标：较小值偏黄，较大值偏紫；星形和初始区域使用参考图的粉红色。
- 热图色块采用实算网格值，10%近优轮廓使用网格插值。没有将插值当作新的数值求解。

运行：`python generate.py`。输入输出均相对脚本目录定位。

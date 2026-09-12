from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Polygon,FancyArrowPatch
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Microsoft YaHei','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14,4))
fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,14),ylim=(0,4));ax.axis('off')
def box(x,y,w,h,text,rounded=False):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f'round,pad=0.02,rounding_size={.22 if rounded else .025}',fc='white',ec='#111111',lw=1.4))
 ax.text(x,y,text,ha='center',va='center',fontsize=17,linespacing=1.5,color='#111111')
def arrow(points):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],c='#111111',lw=1.3)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=14,lw=1.3,color='#111111'))
y=2.05
box(1.05,y,1.8,1.35,'原点出发\n扫描必要频道',True)
box(3.5,y,2.3,1.62,'更新频道状态\n可行区域与\n接收半径区间\nBayes 位置权重')
ax.add_patch(Polygon([(5.8,y+.7),(6.67,y),(5.8,y-.7),(4.93,y)],fc='white',ec='#111111',lw=1.4))
ax.text(5.8,y,'满足\n完成条件？',ha='center',va='center',fontsize=17,linespacing=1.4)
box(8.0,y,2.05,1.62,'生成候选任务\n已发现源：\n补测或清除\n未知频道覆盖检测')
box(10.42,y,2.0,1.35,'预测反馈与\n后续费用\n联合规划访问路线')
box(12.75,y,2.0,1.35,'比较预计完成时间\n选择并执行\n下一步动作')
box(5.8,3.55,1.75,.62,'结束任务',True)
for a,b in [(1.97,2.32),(4.67,4.91),(6.69,6.95),(9.05,9.4),(11.44,11.73)]:arrow([(a,y),(b,y)])
arrow([(5.8,2.77),(5.8,3.21)])
ax.text(5.99,2.98,'是',fontsize=15,va='center')
ax.text(6.82,2.24,'否',fontsize=15,ha='center')
arrow([(12.75,1.35),(12.75,.4),(3.5,.4),(3.5,1.21)])
ax.text(8.0,.63,'根据真实检测或清除反馈更新',fontsize=16,ha='center',va='center',color='#111111')
for ext in ['png']:fig.savefig(OUT/f'q3_bayes_flowchart_bw_horizontal.{ext}',dpi=400,facecolor='white')

import sys
from pycm import *
y_actu = [2, 2, 2,2,2,2,2,2,2,0]
y_pred = [0, 0, 2,2,4,0,0,0,0,2]
cm = ConfusionMatrix(y_actu, y_pred,digit=5)
print(cm)


# Overall MCC
# F1 Macro
# TPR Macro
# PPV_Macro

# 计算性能指标
print("准确率:", cm.Overall_ACC)
# print("召回率:", cm.Recall[1])
print("F1值:", cm.F1_Macro)

# 绘制混淆矩阵图
cm.plot(cmap="Blues")
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sympy.physics.control.control_plots import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# 显示中文字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_csv('E:/RS/LBL/70%.csv')

print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:].values  # 特征列
y = data.iloc[:, 0].values   # 标签列

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 初始化 TabNetClassifier
model = TabNetClassifier()

# 初始化交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存储每次交叉验证的TPR和FPR
auc_scores = []
sensitivities = []
specificities = []
ppv_scores = []
npv_scores = []
accuracy_scores = []
f1_scores = []
fprs = []
roc_auc_scores = []
tprs = []

for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    model.fit(
     X_train=X_train, y_train=y_train,
     eval_set=[(X_test, y_test)],
     eval_name=['test'],
     eval_metric=['accuracy'],
     max_epochs=100,
     patience=20,
     batch_size=32,
     virtual_batch_size=128,
     num_workers=0,
     drop_last=False,
     weights=1
     )

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_scores.append(roc_auc_score(y_test, y_pred_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))
    ppv_scores.append(tp / (tp + fp))
    npv_scores.append(tn / (tn + fn))
    accuracy_scores.append((tp + tn) / (tp + tn + fp + fn))
    f1_scores.append(2 * tp / (2 * tp + fp + fn))

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 绘制热力图
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 4},
                fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Predicted Label', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.title('Confusion Matrix Heat Map', fontsize=8)
    plt.show()

     # 预测概率
    y_score = model.predict_proba(X_test)[:, 1]
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

    # 存储TPR和FPR用于绘制平均ROC曲线
    tprs.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
    fprs.append(np.linspace(0, 1, 100))

# 计算平均ROC曲线
mean_fpr = np.mean(fprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)
mean_roc_auc = auc(mean_fpr, mean_tpr)

# 计算标准差
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 绘制平均ROC曲线和标准差线
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_roc_auc, lw=2, alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

# 绘制对角线（随机猜测）
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8, label='Chance')

# 设置图例和坐标轴标签
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for 5-fold Cross-Validation')
plt.legend(loc="lower right")
plt.show()

# 输出每次交叉验证的评估指标
for i in range(5):
    print(f"第{i + 1}次交叉验证结果：")
    print(f"AUC值：{auc_scores[i]:.2f}")
    print(f"敏感性：{sensitivities[i]:.2f}")
    print(f"特异性：{specificities[i]:.2f}")
    print(f"阳性预测值(PPV)：{ppv_scores[i]:.2f}")
    print(f"阴性预测值(NPV)：{npv_scores[i]:.2f}")
    print(f"准确率：{accuracy_scores[i]:.2f}")
    print(f"F1得分值：{f1_scores[i]:.2f}")
    print()

# 输出最终平均结果
print("最终平均结果：")
print(f"AUC值：{np.mean(auc_scores):.2f}")
print(f"敏感性：{np.mean(sensitivities):.2f}")
print(f"特异性：{np.mean(specificities):.2f}")
print(f"阳性预测值(PPV)：{np.mean(ppv_scores):.2f}")
print(f"阴性预测值(NPV)：{np.mean(npv_scores):.2f}")
print(f"准确率：{np.mean(accuracy_scores):.2f}")
print(f"F1得分值：{np.mean(f1_scores):.2f}")

#____________PR曲线
from sklearn.metrics import precision_recall_curve

# 计算PR曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# 计算AUC-PR
auc_pr = auc(recall, precision)

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, color='blue', label='PR curve (AUC = %0.2f)' % auc_pr)
# 添加随机猜测对角线
plt.plot([0, 1], [1, 0], linestyle='--', color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

from joblib import dump
dump(model, 'E:/RS/LBL/LBL1.pkl')
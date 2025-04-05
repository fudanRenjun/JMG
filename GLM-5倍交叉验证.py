import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
import statsmodels.api as sm
from numpy import interp
import seaborn as sns

# 加载CSV文件
df = pd.read_csv('E:/RS/LBL/70%.csv')

# 初始化五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化保存指标的字典
metrics = {'AUC': [], 'Accuracy': [], 'F1': [], 'PPV': [], 'NPV': [], 'Sensitivity': [], 'Specificity': []}

# 初始化ROC和PR曲线相关变量
tprs = []
mean_fpr = np.linspace(0, 1, 100)
aucs = []
mean_precision = np.linspace(0, 1, 100)
pr_curves = []
pr_aucs = []

# 交叉验证
fold_number = 1

# 存储每折的指标
fold_metrics = []

for train_index, test_index in kf.split(df):
    X_train, X_test = df.iloc[train_index, 1:], df.iloc[test_index, 1:]
    y_train, y_test = df.iloc[train_index, 0], df.iloc[test_index, 0]

    # 使用GLM模型进行拟合
    model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    predictions = model.predict(sm.add_constant(X_test))
    predictions_class = (predictions > 0.5).astype(int)

    # 计算指标
    auc_value = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions_class)
    f1 = f1_score(y_test, predictions_class)
    ppv = precision_score(y_test, predictions_class)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_class).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    # 存储每一折的指标
    metrics['AUC'].append(auc_value)
    metrics['Accuracy'].append(accuracy)
    metrics['F1'].append(f1)
    metrics['PPV'].append(ppv)
    metrics['NPV'].append(npv)
    metrics['Sensitivity'].append(sensitivity)
    metrics['Specificity'].append(specificity)

    # 存储每折的指标值
    fold_metrics.append({
        'Fold': fold_number,
        'AUC': auc_value,
        'Accuracy': accuracy,
        'F1': f1,
        'PPV': ppv,
        'NPV': npv,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, predictions_class)

    # 绘制热力图
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 4},
                fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Predicted Label', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.title(f'Confusion Matrix Heat Map for Fold {fold_number}', fontsize=8)
    plt.show()

    # 绘制每一折的ROC曲线
    fpr, tpr, _ = roc_curve(y_test, predictions)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # 绘制每一折的PR曲线
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    pr_auc = average_precision_score(y_test, predictions)
    pr_curves.append(interp(mean_precision, recall[::-1], precision[::-1]))
    pr_aucs.append(pr_auc)

    fold_number += 1

# 计算平均ROC曲线和标准差
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

# 计算上下限
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

# 绘制平均ROC曲线
plt.figure(figsize=(10, 6))
for i in range(len(tprs)):
    plt.plot(mean_fpr, tprs[i], lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {aucs[i]:.2f})')
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f ± 1 std. dev.)' % mean_auc, lw=2, alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8, label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 5-fold Cross-Validation')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()

# 绘制平均PR曲线
mean_precision = np.mean(pr_curves, axis=0)
mean_pr_auc = auc(np.linspace(0, 1, 100), mean_precision)

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 1, 100), mean_precision, color='b', label=r'Mean PR (AUC = %0.2f)' % mean_pr_auc, lw=2, alpha=.8)
plt.plot([0, 1], [1, 0], linestyle='--', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for 5-fold Cross-Validation')
plt.legend(loc="lower left")
plt.grid(False)
plt.show()

# 输出每次交叉验证的指标
print("Metrics for each fold:")
for fm in fold_metrics:
    print(f"Fold {fm['Fold']}: AUC = {fm['AUC']:.3f}, Accuracy = {fm['Accuracy']:.3f}, "
          f"F1 = {fm['F1']:.3f}, PPV = {fm['PPV']:.3f}, NPV = {fm['NPV']:.3f}, "
          f"Sensitivity = {fm['Sensitivity']:.3f}, Specificity = {fm['Specificity']:.3f}")

# 最终输出每个指标的平均值
print("\nAverage Metrics:")
for metric in metrics:
    print(f"{metric}: {np.mean(metrics[metric]):.3f}")


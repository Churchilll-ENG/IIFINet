import torch
import numpy as np
from h5py.h5f import ACC_RDONLY
from numpy.ma.extras import average
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    F = 0
    A = 0
    total_preds = []
    total_truths = []
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
            A = A + acc
            F = F + f1
            total_preds.extend(test_preds_i)
            total_truths.extend(test_truth_i)
        overall_acc = accuracy_score(total_truths, total_preds)
        print("--Overall ACC:", overall_acc)
        print("--ACC:", A/4)
        print("--F1 :", F/4)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)



def eval_mosei_senti_per_class1(results, truths, num_classes=7, exclude_zero=False):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # 将预测结果和真实标签转换为一维 NumPy 数组
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()


    # 将预测值和真实值裁剪到指定范围（根据类别数）
    if num_classes == 7:
        test_preds = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth = np.clip(test_truth, a_min=-3., a_max=3.)
    elif num_classes == 5:
        test_preds = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth = np.clip(test_truth, a_min=-2., a_max=2.)

    # 离散化预测值（四舍五入并确保落在指定范围内）
    test_preds = np.round(test_preds)
    test_truth = np.round(test_truth)

    # 计算每个类别的单独准确率、样本数量和 F1 分数
    per_class_accuracy = {}
    per_class_count = {}
    per_class_f1 = {}
    unique_labels = np.unique(test_truth)  # 确定类别

    for label in unique_labels:
        label_indices = test_truth == label  # 筛选该类别的索引
        class_acc = accuracy_score(test_truth[label_indices], test_preds[label_indices])  # 准确率
        sample_count = label_indices.sum()  # 样本数量

        # 计算 F1 分数
        class_f1 = f1_score(test_truth[label_indices], test_preds[label_indices], average='weighted')

        # 存储结果
        per_class_accuracy[label] = class_acc
        per_class_count[label] = sample_count
        per_class_f1[label] = class_f1

    # 打印每个类别的准确率、样本数量和 F1 分数
    for label in unique_labels:
        acc = per_class_accuracy[label]
        count = per_class_count[label]
        f1 = per_class_f1[label]
        print(f"Class {label}: Accuracy = {acc:.4f}, Samples = {count}, F1 Score = {f1:.4f}")


    return per_class_accuracy, per_class_count, per_class_f1


def eval_single(results, truths,hyp_params, single=-1):
    emos1 = ["fea", "hap", "sad", "ang", "dis", "sur"]
    emos = ["fear", "happiness", "sadness", "angry", "disgust", "surprise"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        for i in range(6):
            if hyp_params.dataset == emos1[i]:
                emos[1] = emos[i]
                break
        print(f"{emos[1]}: ")
        test_preds_i = np.argmax(test_preds[:, 1], axis=1)
        test_truth_i = test_truth[:, 1]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)

        # for emo_ind in range(1):
        #     print(f"{emos[emo_ind]}: ")
        #     test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        #     test_truth_i = test_truth[:, emo_ind]
        #     f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        #     acc = accuracy_score(test_truth_i, test_preds_i)
        #     print("  - F1 Score: ", f1)
        #     print("  - Accuracy: ", acc)

    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


def eval_new(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    F = 0
    A = 0
    if single < 0:
        test_preds = results.view(-1, 2, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 2).cpu().detach().numpy()

        for emo_ind in range(2):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
            print(len(test_preds_i))
            A = A + acc
            F = F + f1
        print("--ACC:", A / 4)
        print("--F1 :", F / 4)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


def save_confusion_matrix(cm, labels, title="Confusion Matrix", save_path="confusion_matrix.png"):
    """
    生成并保存混淆矩阵的热力图
    :param cm: 混淆矩阵
    :param labels: 标签列表
    :param title: 图表标题
    :param save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    # 显示为热力图，annot为True表示显示数值，fmt='g'表示按照整数格式显示
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    # 绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.4f', cmap='binary', xticklabels=labels, yticklabels=labels)
    # sns.heatmap(cm, annot=True, fmt='.2f', cmap='binary', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(rotation=0)  # 设置 x 轴文字水平显示
    plt.yticks(rotation=0)  # 设置 y 轴文字水平显示（默认水平）
    plt.tight_layout()  # 保证显示不被裁剪
    plt.savefig(save_path)  # 保存图片
    plt.close()  # 关闭以释放内存


def eval_iemocap1(results, truths, save_dir="./confusion_matrices"):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    emos1 = ["Ha", "Ne", "An", "Sa"]
    total_preds = []
    total_truths = []
    # 将结果和真实标签转换为numpy数组
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()

    for i in range(len(test_preds)):
        test_preds_i = np.argmax(test_preds[i], axis=1)  # 预测标签
        differences = test_preds[i][:, 1] - test_preds[i][:, 0]

        # 找到最大差值的索引
        max_diff_index = np.argmax(differences)
        test_truth_i = test_truth[i]  # 真实标签
        if max_diff_index == 0:
            total_preds.append(0)
        elif max_diff_index == 1:
            total_preds.append(1)
        elif max_diff_index == 2:
            total_preds.append(2)
        elif max_diff_index == 3:
            total_preds.append(3)
        else:
            total_preds.append(0)
        if test_truth_i[0] == 1:
            total_truths.append(0)
        if test_truth_i[1] == 1:
            total_truths.append(1)
        if test_truth_i[2] == 1:
            total_truths.append(2)
        if test_truth_i[3] == 1:
            total_truths.append(3)
    cm_overall = confusion_matrix(total_truths, total_preds, labels=[0, 1, 2, 3])
    # cm_overall[[0, 1]] = cm_overall[[1, 0]]
    cm_overall[[2, 3]] = cm_overall[[3, 2]]
    # cm_overall[:, [0, 1]] = cm_overall[:, [1, 0]]
    cm_overall[:, [2, 3]] = cm_overall[:, [3, 2]]
    cm_overall[[0, 3]] = cm_overall[[3, 0]]
    cm_overall[:, [0, 3]] = cm_overall[:, [3, 0]]
    overall_acc = accuracy_score(total_truths, total_preds)
    f1 = f1_score(total_truths, total_preds, average='weighted')
    f1_un = f1_score(total_truths, total_preds, average='macro')
    print(f"--Overall ACC: {overall_acc:.4f}")
    print(f"--Overall F1 Score: {f1}")
    print(f"--Overall F1 Un Score: {f1_un:.4f}")

    # 保存整体混淆矩阵
    print("Overall Confusion Matrix:")
    overall_save_path = f"{save_dir}/overall_confusion_matrix.png"
    save_confusion_matrix(cm_overall, emos1, title="Overall Confusion Matrix", save_path=overall_save_path)

    return cm_overall  # 返回混淆矩阵以便进一步使用（如统计分析）





#%%
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def pair_plot(pairs, labels, labels2 ,to_show=6, num_col=4, predictions=None, test=False):
    '''
    함수 목적 : 이미지 쌍에 대한 시각화
    인풋:
        - pairs : 이미지 쌍(np.array)
        - labels : id 레이블(np.array)
        - labels2 : 성별 레이블(np.array)
        - to_show : 시각화 할 이미지 수(int)
        - num_col : 한 행에 보여질 이미지 열 수(int)
        - predictions : 모델 적용한 예측 값 배열(np.array)
        - test : 시각화 할 데이터의 train/test set 여부(bool)     
    '''

    num_row = to_show // num_col if to_show // num_col != 0 else 1

    to_show = num_row * num_col

    fig, axes = plt.subplots(num_row, num_col, figsize=(8, 8))
    for i in range(to_show):
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        
        if test: # 동일인 판별 - True : 실제 값(0:비동일인, 1:동일인), Pred : 예측 값 / 성별 판별 - True: 실제 값(0:남성, 1:여성), Gender : 예측 값 
            ax.set_title("True: {} | Face: {:.2f} \n True: {} | Gender: {:.2f}".format(labels[i], predictions[0][i][0],labels2[i],predictions[1][i][0]), size=8)
        else: # Face : 실제 동일인 여부(0:비동일인, 1:동일인), Gender : 실제 성별(0:남성, 1:여성)
            ax.set_title("Face: {} | Gender: {}".format(labels[i],labels2[i]), size=8)
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    #%%
    plt.show();



def cm_plot(labelTest, predictions, type):
    '''
    함수 목적 : 예측 결과에 따른 confusion matrix 시각화 및 accuracy, precision, recall, f1 score 확인
    인풋 :
        labelTest : 실제 라벨 값 배열(np.array)
        predictions : 예측 값 배열(np.array)
        type : 동일인 판별, 성별 판별 중 어떤 예측값에 대한 confusion matrix 인지(str, 'face' or 'sex')
    '''
    y_pred = np.where(predictions>0.5,1,0)
    cm = confusion_matrix(labelTest, y_pred)
    if type == 'face':
        labels = ['Not Same', 'Same']
    else:
        labels = ['Male','Female']
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    disp.plot(cmap = plt.cm.Blues)
    plt.title(f'{type}')
    #%%
    plt.show();
    print(f'[ {type} report ]')
    print(classification_report(labelTest, y_pred))
    print('Accuracy score : {0:.2f}'.format(accuracy_score(labelTest, y_pred)))
    print('Precision score : {0:.2f}'.format(precision_score(labelTest, y_pred)))
    print('Recall score : {0:.2f}'.format(recall_score(labelTest, y_pred)))
    print('F1 score : {0:.2f}'.format(f1_score(labelTest, y_pred)))


def plot_history(history):
    '''
    함수 목적 : 학습한 모델의 Epoch에 따른 accuracy, loss 값 시각화
    인풋 :
        - history : model.fit하여 나온 history
    '''
    history_df = pd.DataFrame(history.history)
    history_df.plot(figsize=(12,8))

    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Variable')
    #%%
    plt.show();
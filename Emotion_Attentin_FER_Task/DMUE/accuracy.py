import pandas as pd
# reads the csv, takes only the first column and creates a set out of it.
gt_forward = pd.read_csv("/home/ubuntu/ksy/SNU/dataset/all.csv",
        index_col=False, header=None)[5]
gt_name = pd.read_csv("/home/ubuntu/ksy/SNU/dataset/all.csv",
        index_col=False, header=None)[1]
inf_forward = pd.read_csv("/home/ubuntu/ksy/SNU/DMUE/deep_face_results.csv", index_col=False,  # [alls_results.csv, deepface_labels_results.csv]
                    header=None)[1]  # same here
inf_name = pd.read_csv("/home/ubuntu/ksy/SNU/DMUE/deep_face_results.csv", index_col=False,
                    header=None)[0]  # same here
gt_forward_list = list(gt_forward)
gt_name_list = list(gt_name)
inf_forward_list = list(inf_forward)
inf_name_list = list(inf_name)
count_1 = 0
count_2 = 0
count_3 = 0
len_1=0
len_2=0
len_3=0
miss_label = 0

# 감정상태 정확도 측정 코드
for i in range(0, len(inf_name_list)):
    gt_index = gt_name_list.index(inf_name_list[i])

    if gt_forward_list[gt_index] == '1.0':
            len_1+=1
            if inf_forward_list[i]==1:
                count_1 += 1
    elif gt_forward_list[gt_index] == '2.0':
            len_2+=1
            if inf_forward_list[i] == 2:
                count_2 += 1
    elif gt_forward_list[gt_index] == '3.0':
            len_3+=1
            if inf_forward_list[i] == 3:
                count_3 += 1
    else:
        miss_label+=1

print(len_1, len_2, len_3, miss_label)
print(count_1,count_2,count_3)
total = count_1+count_2+count_3
print("1 accuracy : {:.2f}%".format(count_1/len_1*100))
print("2 accuracy : {:.2f}%".format(count_2/len_2*100))
print("3 accuracy : {:.2f}%".format(count_3/len_3*100))
print("total accuracy : {:.2f}%".format(total/(len_1+len_2+len_3)*100))


from super_class_neural_0_0_3 import *
import time
#setting
SELF_LEARNING = False

DATA_CLESS=[
[15,60,155,4],
[52,50,170,4],
[30,75,173,2],
[22,66,180,3],
[17,69,163,2],
[19,45,153,3],
[66,65,165,2],
[39,80,163,4],
[10,44,146,2],
[35,80,150,2],
[100,64,158,3],
[10,45,145,2]
]
DATA_VALUE=[1,2,1,2,1,2,1,1,2,1,1,3]

predict_value = [10,50,150,2]

data_value_unduplicate = list(np.unique(np.array(DATA_VALUE)))
DATA_VALUE_DIGEST = [[0]*len(DATA_VALUE) for _ in data_value_unduplicate]
for index , value in enumerate(DATA_VALUE):
    DATA_VALUE_DIGEST[data_value_unduplicate.index(value)][index] = 1
# print(DATA_VALUE_DIGEST)
scn = super_class_neural(scope=1)
data = scn.customize_model(DATA_CLESS,DATA_VALUE)
r_weights = scn.change(DATA_CLESS,data)
test = scn.change([predict_value],data)

predict = []

time_start = time.time()
for value in DATA_VALUE_DIGEST:
    list_test = list(test)
    weigths = Weigths(r_weights,value)
    pre = weigths.F_predict(list_test[0])
    predict.append(pre)
time_end = time.time()
print("time:",time_end-time_start)
# print(predict)
print("Type:",data_value_unduplicate[predict.index(max(predict))])
# print(pre)
# if pre >= 0.56:print(1,round(pre*100, 2))
# elif pre <= 0.55:print(0,round(pre*100, 2))
# print("time:",time_end-time_start) #0.78
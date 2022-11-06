from weights_neural import *
import numpy as np
#setting
class super_class_neural :

    def __init__(self, scope=1,fine_scope_information=False,scpoe_leve_class=2):
        self.SCOPE = scope
        self.FIND_SCOPE_INFORMATION = fine_scope_information #True or False
        self.SCOPE_LEVE_CLASS = scpoe_leve_class * self.SCOPE #0.0 - 1.0
    
    def most_frequent(self,List):
        counter = 0
        num = List[0]
        
        for i in List:
            curr_frequency = List.count(i)
            if(curr_frequency> counter):
                counter = curr_frequency
                num = i
    
        return num
    def customize_model(self,DATA,DATA_VALUE):
        
        DATA_CLESS = DATA
        DATA = np.array(DATA_CLESS)
        data_sci_result = []
        key_sci_result = []
        for value_len in range(len(DATA[0])):
            in_class = DATA[:,value_len]
            sum_ = np.sum(in_class)
            max_ = np.max(in_class)
            min_ = np.min(in_class)
            # print(value_len,max_,min_)

            n_ = int(len(DATA_CLESS)*self.SCOPE_LEVE_CLASS)+1
            leve_class = (sum_/n_)/(max_-abs(min_))
            leve_high = min_+(leve_class*n_)

            class_leve_data = []
            sum_leve_class = min_

            while sum_leve_class <= (max_+leve_class):
            # for value in range(0,n_):
                class_leve_data.append( sum_leve_class )
                sum_leve_class += leve_class
            # class_leve_data = [* range(min_ , int(leve_high+1) , int(leve_class+1) )]
            arr_data_compare = np.array([None]*(len(class_leve_data)))

            top_in_class = class_leve_data[-1]
            top_max_value_arr_class = 0
            _in_class = in_class.tolist()
            at  =0
            for index, value in enumerate(class_leve_data[::-1]):
                # arr_data_compare[index] = [[] ,[],[top_in_class,value]]
                value_arr_class = in_class[(in_class<=top_in_class) & (in_class>=value)]
                if len(value_arr_class) != 0:
                    if len(value_arr_class) > top_max_value_arr_class: top_max_value_arr_class = len(value_arr_class)

                    mostly = []
                    arr_vac_index = 0
                    at += 1
                    for vac in value_arr_class :
                        vac_index = _in_class.index(vac)
                        _in_class[vac_index] = None
                        # _in_class[vac_index] = 999
                        arr_vac_index += 1
                        mostly.append(DATA_VALUE[vac_index])
                        # print("_in_class[vac_index]",_in_class[vac_index])

                    dup = self.most_frequent(mostly)
                    point = [1,[top_in_class,value],mostly,mostly,arr_vac_index,value_arr_class]
                    # print(point)
                else: point = [0,[top_in_class,value],[0],[0],[0],[0]]
                arr_data_compare[index] = point
                top_in_class = value
            arr_data_compare = arr_data_compare[::-1]
            key_sci_result.append( top_max_value_arr_class )
            data_sci_result.append( arr_data_compare )

        # model_weights = DATA
        arr_weights = data_sci_result
        if self.FIND_SCOPE_INFORMATION:
            # print(data_sci_result)
            arr_weights = [[] for _ in " "*len(data_sci_result)]
            for index_asm,arr_sci_model in enumerate(data_sci_result):
                for dsr in data_sci_result[index_asm]:
                    if dsr[0] != None:
                        arr_weights[index_asm].append(dsr)
        return (arr_weights,key_sci_result)

    def change(self,model_weights_t,result):
            arr_weights_t,key_sci_result = result
            data_change = []
            key_change = []
            model_weights_t = np.array(model_weights_t)
            for value_len in range(len(model_weights_t[0])):
                da_ch_in_arr= [] 
                key_change_arr= []
                rekey = [0]*key_sci_result[value_len]
                # for index_a_w,a_w in enumerate(model_weights_t):
                for index_a_w,a_w in enumerate(arr_weights_t):
                    for index_arr,arr in enumerate(a_w):
                        key = int("1"+str(value_len)+str(index_a_w)+str(index_arr))
                        arr[2] = [k for k in (arr[2]+rekey)[0:key_sci_result[value_len]+1]]
                        arr_all = np.array([[arr[0]]+arr[2]]).flatten()
                        da_ch_in_arr.append(arr_all)
                        key_change_arr.append( key )
                        model_weights_t[ (model_weights_t <= arr[1][0]) & (model_weights_t >= arr[1][1])] = key
                        # print(model_weights_t[ (model_weights_t <= arr[1][0]) & (model_weights_t >= arr[1][1])])
                        # data = np.where(( (model_weights_t <= arr[1][0]) & (model_weights_t >= arr[1][1]) ), 0, arr_all)
                key_change.append(key_change_arr)
                data_change.append(da_ch_in_arr)
                # print("data_change",data_change)
            # print(model_weights_t,data)
            r_weights = [model_weights_t,arr_weights_t,data_change,key_change]

            model_weights_data = r_weights[0]
            data_change = list(np.array(r_weights[2], dtype=object).flatten())
            key_change = list(np.array(r_weights[3]).flatten())

            data_weights_neural = [[] for _ in "_"*len(model_weights_data)]
            for index_rkm, return_key_model in enumerate(model_weights_data):
                for index_r_key, r_key in enumerate(return_key_model):
                    key = key_change.index(r_key)
                    data_from_key = data_change[key]
                    data_weights_neural[index_rkm] = np.append(data_weights_neural[index_rkm],data_from_key)

                    data_weights_neural[index_rkm] = np.array(data_weights_neural[index_rkm]).flatten()
            return data_weights_neural
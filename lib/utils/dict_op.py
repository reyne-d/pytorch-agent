import numpy as np
import torch

def slice_dic(dic, index):
    ret_dic = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            ret_dic[k] = slice_dic(v, index)
        elif isinstance(v, np.ndarray):
            ret_dic[k] = v[index]
        else:
            raise ValueError("{}:{} is neither dic nor np.ndarray".format(k,v))
    return ret_dic


def flatten_dic(dic):
    kv_list = []
    for k, v in dic.items():
        if isinstance(v, dict):
            tmp_dic = flatten_dic(v)
            for tk, tv in tmp_dic.items():
                kv_list.append(['{}.{}'.format(k, tk), tv])
        else:
            kv_list.append([k, v])
    ret_dic = {k: v for (k, v) in kv_list}
    return ret_dic

def concat_dic(dic_list):
    """
        Concat list of dict to one dict.
    :param dic_list: list of dict object,  each elements in list should have the same format(keys)
    :return:
    """
    assert all([isinstance(v, dict) for v in dic_list]), "elements are expected dic"
    assert all([dic_list[0].keys()==v.keys() for v in dic_list]), "elements are expected have keys"
    cat_dic = {k:[] for k in dic_list[0].keys()}
    for dic in dic_list:
        for k,v in dic.items():
                cat_dic[k].append(v)

    for k,v  in cat_dic.items():
        if isinstance(v[0], np.ndarray) or isinstance(v[0], list):
            cat_dic[k] = np.concatenate(v)
        if isinstance(v[0], torch.Tensor):
            cat_dic[k] = torch.cat(v).to(torch.device("cpu")).numpy()
    return cat_dic


if __name__=='__main__':

    # test slice_dic
    dic = {
        'a': np.arange(5),
        'b': {
            'b1': np.arange(5),
            'b2': np.arange(5),
            'c': {
                'c1': np.arange(5),
                'c2':np.arange(5)
            }
        }
    }
    print ("original dic: ", dic)
    dic = slice_dic(dic, index=[2,3])
    print("slice with index[2,3]: ", dic)

    # test flatten_dic
    dic = flatten_dic(dic)
    print("flatten:", dic)

    # test concaten
    dic_list = [{'a':[1,2,3],'b':np.arange(5),'c':['aa','bb','cc'], 'd': 'aaa'},
                {'a': [1, 2, 3], 'b': np.arange(5), 'c': ['aa', 'bb', 'cc'], 'd': 'aaa'},
                {'a': [1, 2, 3], 'b': np.arange(5), 'c': ['aa', 'bb', 'cc'], 'd': 'aaa'}]
    print("dic_list: {}".format(dic_list))
    cat_dic = concat_dic(dic_list)
    print('cat_dic : {}'.format(cat_dic))

def list_from_file(filename,prefix='',offset=0,max_num=0):
    """
    max_num:需要读的最大的行数，小于0时不起作用

    return :类别字符串组成的list
    """
    cnt = 0
    item_list = []
    with open(filename,'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num>0 and cnt>max_num:
                break
            item_list.append(prefix+line.rstrip('\n'))
            cnt+=1
    return item_list

if __name__=="__main__":
    print(list_from_file('test_class.txt'))
file1 = 'train.txt'
file2 = 'test.txt'
import pickle
import pickle
def merge(file1, file2):
    f1 = open(file1, 'a+')
    with open(file2, 'r') as f2:
        f1.write('\n')
        for i in f2:
            f1.write(i)
 
 
train_data = pickle.load(open(file1, 'rb'))
print(train_data)
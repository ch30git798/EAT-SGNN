import pickle
A = pickle.load(open('train.txt','rb'))
print(A[:10])
print(len(A))
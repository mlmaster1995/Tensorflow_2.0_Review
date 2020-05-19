from WKNN import WKNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
print('data sample: \n', iris['data'][0:10,:])
print('target sample: \n', iris['target'][0:10])
data, test_data, labels, test_labels = train_test_split(iris['data'], iris['target'], test_size=0.3, random_state=1)
print('data size: ', len(data))
print('label size: ', len(labels))
print('test data size: ', len(test_data))
print('test label size: ', len(test_labels))

wknn = WKNN(train_data=data, train_labels=labels, k_value=10)
predict_res = wknn.predict(test_data)
print('test data prediction: ', predict_res)

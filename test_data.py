from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
print('data: \n', iris['data'][0:10,:])
print('target: \n', iris['target'][0:10])
data, test_data, labels, test_labels = train_test_split(iris['data'], iris['target'], test_size=0.3, random_state=1)
print('data size: ', len(data))
print('label size: ', len(labels))
print('test data size: ', len(test_data))
print('test label size: ', len(test_labels))
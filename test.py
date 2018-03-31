import helper
import submission
from sklearn.metrics import f1_score

training_data = helper.read_data('./asset/training_data.txt')
#training_data = helper.read_data('./asset/tiny_train.txt')
classifier_path = './asset/classifier.dat'
submission.train(training_data, classifier_path)
test_data = helper.read_data('./asset/tiny_test1.txt')
prediction = submission.test(test_data, classifier_path)
#print(prediction)
#ground_truth = [1,1,2,1]
ground_truth = [1,1,1,1,2,1,1,2,1,1,2,1,1,2,2,2,2,1,1,1,3,2]
print(f1_score(ground_truth, prediction, average='macro'))

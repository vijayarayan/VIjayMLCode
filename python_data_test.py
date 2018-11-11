
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE) 
x = list(reader)
print(x)
data = numpy.array(x).astype('float')
print("data is %s" %data)
print(data.shape)
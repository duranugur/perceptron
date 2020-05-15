#import libraries
import numpy as np #veri üzerinde matematiksel işlem yapmak için
import csv
import matplotlib.pyplot as plt #verileri grafiksel tabloda göstermek için

# notice that you should run it only with python3

class Perceptron(object):
    #değer atama işlemleri.
    def __init__(self, no_of_inputs, learning_rate=0.5, bios_learning_rate=0.5):
        self.learning_rate = learning_rate
        self.bios_learning_rate = bios_learning_rate
        self.weights = [0.1,0.2,0.3]
    #her değer çağırdığımızda bu method calısacak.
    def getter(self):
        return self.weights
    #burada atama işlemleri yapılıyor.aşağıda array olarak tanımlanan trainin_input'un
    #sıfırıncı ve birinci değerleri grup1_x 'e atanıyor.
    def train(self, training_inputs, target):
        # print(training_inputs)
        group1_x: object  = training_inputs[0]
        x2 = training_inputs[1]
        b = self.weights[2]
        w1 = self.weights[0]
        w2 = self.weights[1]
        y = b + group1_x * w1 + x2 * w2 # y sonucu hesaplanıyor.

        #perceptron sonucu 0'dan büyükse y=1.0 olarak işlem görüyor.değilse 0.0.
        if y > 0:
            y = 1.0
        else:
            y = 0.0 #eksi değer önlemek için 0.0 'a eşitliyoruz.

        #burada değerler yeniden güncelleniyor. her veri icin güncelleniyor.
        self.weights[2] = b + self.bios_learning_rate * (target - y)
        self.weights[0] = w1 + self.learning_rate * (target - y) * group1_x
        self.weights[1] = w2 + self.learning_rate * (target - y) * x2
        return y

#
network = Perceptron(2)
training_inputs = []
label = []
group0_x = []
group0_y = []
group1_x = []
group1_y = []
epochs = 5000
num_of_miss = {}
accuracy_of_each_epoch = {}
miss = 0

#
for i in range(epochs):
    #verileri okudugumuz data.csv dosyanını okuyor.
    with open("data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        #datadan gelen her değerin x ve y değerlerini okuyor.
        for row in readCSV:
            training_inputs = [float(row[0]),float(row[1])]
            label = float(row[2])
            #prediction label'dan O dönerse  miss 1 arttırılıyor.
            prediction = network.train(training_inputs, label)
            if prediction != label:
                miss += 1
            #training_input'dan dönen değerleri label'da hesaplıyor.
            if label == 1.0:
                group1_x.append(float(row[0]))
                group1_y.append(float(row[1]))
            if label == 0.0:
                group0_x.append(float(row[0]))
                group0_y.append(float(row[1]))
        #her miss değeri arttıgında lineer olarak accuracy epoch değerlerini array'de tutuyor.
        num_of_miss[i] = miss
        accuracy_of_each_epoch[i] = ((200-num_of_miss[i])/200)*100
        print("accuracy of epoch %d -----> %f " %(i,accuracy_of_each_epoch[i]))
        print("loss of epoch %d is %f " %(i,(((num_of_miss[i])/200.0)*100)))
        print()
        miss = 0

t = []
for i in range(epochs):
    # print(num_of_miss[i])
    t.append(float(num_of_miss[i])/200.0)

weights = network.getter()
w1 = weights[0]
w2 = weights[1]
b = weights[2]

# Create data
x_g0 = np.array(group0_x)
y_g0 = np.array(group0_y)
x_g1 = np.array(group1_x)
y_g1 = np.array(group1_y)
g0 = (x_g0,y_g0)
g1 = (x_g1, y_g1)
data = (g0, g1)

colors = ("blue", "red")
groups = ("Group1", "Group2")
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
 
for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)

a = -w1 / w2
b = -b / w2
lineX = [-200, -60]
lineY = [-200 * a + b, -60 * a + b]

ax.plot(lineX, lineY)
plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
X = np.linspace(0, epochs, len(t))
plt.plot(X, np.array(t))
plt.show()

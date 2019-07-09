import matplotlib as plt
import numpy


weight_value = 1000

#modify to start, where the step function starts
bias_value_1 = 5000

# modify the change where the step function ends
bias_value_2 = -5000

plt.axes([-10,10,-1,10])


inputs = numpy.arrange(-10,10,0.01)
outputs = list()

for x in inputs:
    y1 = 1.0/(1.0+numpy.exp(-weight_value*x-bias_value_1))
    y2 = 1.0/(1.0+numpy.exp(-weight_value*x-bias_value_2))

    # modifiy to change the height of the weight function
    w = 7

    # network output
    y = y1*w -y2 * w

    outputs.append(y)

plt.plot(inputs,outputs, lw = 2, color = 'black')
plt.show()





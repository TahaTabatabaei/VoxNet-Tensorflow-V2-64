import matplotlib.pyplot as plt

# Initialize lists to store the data
x = []
y1 = []
y2 = []

# Read data from the file
with open('checkpoints\\accuracies.txt', 'r') as file:
    for line in file:
        values = line.split()
        x.append(int(values[0]))
        y1.append(float(values[1]))
        y2.append(float(values[2]))

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='train acc', marker='o', linestyle='-', color='r')
plt.plot(x, y2, label='test acc', marker='x', linestyle='-', color='b')

# Adding titles and labels
plt.title('Accuracy Plot')
plt.xlabel('Step')
plt.ylabel('Accuracy')

# Adding a legend
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()

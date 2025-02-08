import matplotlib.pyplot as plt

title = 'Training Loss'
save = True

# Read losses from the text file
losses = []
with open("loss.txt", 'r') as file:
    for line in file:
        try:
            items = line.strip().split(',')
            if items[0] == 'Title':
                title = ' '.join(items[1:])
            loss = float(items[1])
            losses.append(loss)
        except ValueError:
            continue

# Plot the losses if there are any
if losses:
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(title + '.png')
    plt.show()
else:
    print("No data to plot.")


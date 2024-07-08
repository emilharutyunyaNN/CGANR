import matplotlib.pyplot as plt
import os

log_path = './gan_r_losses.log'

G_total = []
G_l1 = []
R_total = []
import re
lists = {
    'Total_loss_G': [],
    'L1_loss_G': [],
    'Total_loss_R': [],
    'Validation_R': []

}
patterns = {
    'Total_loss_G': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: Total loss for G: ([+-]?[0-9]*[.]?[0-9]+(?:[eE][+-]?[0-9]+)?)"),
    'L1_loss_G': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: L1 loss for G: ([+-]?[0-9]*[.]?[0-9]+(?:[eE][+-]?[0-9]+)?)"),
    'Total_loss_R': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: Total loss for R: ([+-]?[0-9]*[.]?[0-9]+(?:[eE][+-]?[0-9]+)?)"),
    'Validation_R': re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: Validation R total loss: ([+-]?[0-9]*[.]?[0-9]+(?:[eE][+-]?[0-9]+)?)")

}
#pattern_g_total = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: Total loss for G: ([0-9.-e]+)")
#pattern_g_l1 = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: L1 loss for G: ([0-9.-e]+)")
#pattern_r_total = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+): INFO: Total loss for R: ([0-9.-e]+)")

with open(log_path, 'r') as file:
    for line in file:
        
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                #print(line)
                #print(match.group(2))
                lists[key].append(float(match.group(2)))


plt.figure(figsize=(10, 6))
print(lists['Total_loss_G'])
print(lists['L1_loss_G'])
print(lists['Total_loss_R'])
# Plot Total_loss_G
plt.plot(lists['Total_loss_G'], label='Total loss for G')

# Plot L1_loss_G
plt.plot(lists['L1_loss_G'], label='L1 loss for G')
plt.title('GAN Losses Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')

# Adding a legend
plt.legend()
plt.savefig('./train_losses_G.jpg')
# Plot Total_loss_R
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(lists['Total_loss_R'], label='Total training loss')


# Adding titles and labels
plt.title('R Losses Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')

# Adding a legend
plt.legend()

# Display the plot
plt.savefig('./losses_R_training.jpg')
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(lists['Validation_R'], label = 'Total validation loss')

# Adding titles and labels
plt.title('R Losses Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')

# Adding a legend
plt.legend()

# Display the plot
plt.savefig('./losses_R_validation.jpg')



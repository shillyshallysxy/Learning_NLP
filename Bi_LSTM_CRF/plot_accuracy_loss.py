import os
import matplotlib.pyplot as plt
import re
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp'
model_log_name = 'model_log.txt'
model_log_path = os.path.join(data_folder_name, data_path_name, model_log_name)

with open(model_log_path, 'r') as f:
    logs = f.readlines()
logs = [log for log in logs if len(log) > 1]
log_dict = {}
for log in logs:
    log_split = log.strip().split(':')
    data = list(map(eval, re.findall(r'\d+.\d+', log_split[1])))
    log_dict[log_split[0]] = data

correct_train_loss = []
for ix, x in enumerate(log_dict['train_loss']):
    if (ix+1) % 10 == 0:
        correct_train_loss.append(x)
log_dict['train_loss'] = correct_train_loss
correct_train_acc = []
for ix, x in enumerate(log_dict['train_acc']):
    if (ix+1) % 10 == 0:
        correct_train_acc.append(x)
log_dict['train_acc'] = correct_train_acc

plt.plot(log_dict['train_loss'], 'k-', label='Train Loss')
plt.plot(log_dict['test_loss'], 'r-', label='Test Loss')
plt.title('Batch Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(log_dict['train_acc'], 'k-', label='Train Acc')
plt.plot(log_dict['test_acc'], 'r-', label='Test Acc')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


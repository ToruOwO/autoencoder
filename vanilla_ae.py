import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# hyper parameters
num_epochs = 50
batch_size = 128
input_size = 784
hidden_size = 30
learning_rate = 0.001
log_per_iter = 100
log_per_epoch = 10
data_dir = './data'
output_dir = './output'

# load MNIST dataset
dataset = datasets.MNIST(root=data_dir,
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True)


class AutoEncoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU()
		)

		self.decoder = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, input_size),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


# set up model and loss function
model = AutoEncoder(input_size, hidden_size)
if torch.cuda.is_available():
	model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):

	print('=' * 20, 'epoch {}'.format(epoch), '=' * 20)

	for i, (img, _) in enumerate(dataloader):
		# convert input tensor to torch Variable
		img = img.view(img.size(0), -1)
		if torch.cuda.is_available():
			img = img.cuda()
		img = Variable(img)

		# forward pass and loss calculation
		output = model(img)
		loss = criterion(output, img)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % log_per_iter == 0:
			print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
			      % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.item()))

	# save images for debugging
	if epoch % log_per_epoch == 0:
		original = img.view(img.size(0), 1, 28, 28)
		reconst = output.view(output.size(0), 1, 28, 28)
		torchvision.utils.save_image(original.data.cpu(), output_dir + '/img_{}.png'.format(epoch))
		torchvision.utils.save_image(reconst.data.cpu(), output_dir + '/img_{}_reconst.png'.format(epoch))

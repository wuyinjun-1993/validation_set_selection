import torch
import torchvision
import os,sys
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.utils.data import DataLoader

from src.common.parse_args import parse_args
from find_valid_set import get_representative_valid_ids_rbc, get_representative_valid_ids_gbc
from models.resnet3 import resnet18


args = parse_args()

valid_count = args.valid_count

transform_train_list = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
transform_train = transforms.Compose(transform_train_list)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(
    root=os.path.join(args.data_dir, 'CIFAR-10'),
    train=True,
    download=True,
    transform=transform_train,
)
testset = torchvision.datasets.CIFAR10(
    root=os.path.join(args.data_dir, 'CIFAR-10'),
    train=False,
    download=True,
    transform=transform_test,
)

net = resnet18(num_classes=10)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

#select valid samples with rbc
valid_ids, new_valid_representations = get_representative_valid_ids_rbc(trainset, criterion, optimizer, trainloader, args, net, valid_count)



#select valid samples with gbc
valid_ids, new_valid_representations = get_representative_valid_ids_gbc(trainset, criterion, optimizer, trainloader, args, net, valid_count)

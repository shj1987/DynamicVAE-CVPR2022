
import torch
import torchvision
import torchvision.transforms as transforms


def return_data(args):
    """docstring for data_loader"""
    image_size = args.image_size
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ])
    ## [0,1]-0.5/0.5 = [-1,1]
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                     shuffle=False, num_workers=4)
    print('***image size >> : ', image_size)
    if args.train:
        return trainloader
    else:
        print('-'*80)
        print("loading testing data now-")
        print('-'*80)
        return testloader
        

    

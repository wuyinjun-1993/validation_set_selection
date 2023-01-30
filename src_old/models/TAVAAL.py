import os, sys

from models.ResNet import resnet34_imagenet
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import random
from exp_datasets.mnist import mnist_to_device
import main.main_train
from exp_datasets.dataloader import dataset_wrapper, split_train_valid_set_by_ids
from models.resnet3 import resnet34
from models.LeNet5 import *
from models.DNN import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset, Dataset, DataLoader, RandomSampler


MARGIN = 1.0
WEIGHT = 1.0
CUDA_VISIBLE_DEVICES = 0
EPOCHL = 120
SUBSET = 10000
MILESTONES = [80, 90]
ADDENDUM = 10
EPOCHV = 10


class LossNet(nn.Module):
    def __init__(
        self,
        feature_sizes=[32, 16, 8, 4],
        num_channels=[64, 128, 256, 512],
        interm_dim=128,
        first = True,
        last = True
    ):
        super(LossNet, self).__init__()
        if first:
            self.GAP1 = nn.AvgPool2d(feature_sizes[0])
            self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        
        if last:
            self.GAP3 = nn.AvgPool2d(feature_sizes[2])
            self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        if first:
            self.FC1 = nn.Linear(num_channels[0], interm_dim)
            self.FC2 = nn.Linear(num_channels[1], interm_dim)
        
        if last:
            self.FC3 = nn.Linear(num_channels[2], interm_dim)
            self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.first = first
        self.last = last
        if not first:
            self.linear = nn.Linear(2 * interm_dim, 1)
        else:
            self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        k = 0
        out_ls = []
        if self.first:
            out1 = self.GAP1(features[k])
            k += 1
            out1 = out1.view(out1.size(0), -1)
            out1 = F.relu(self.FC1(out1))
            out_ls.append(out1)

            out2 = self.GAP2(features[k])
            k += 1

            out2 = out2.view(out2.size(0), -1)
            out2 = F.relu(self.FC2(out2))

            out_ls.append(out2)

        if self.last:
            out3 = self.GAP3(features[k])
            k+=1

            out3 = out3.view(out3.size(0), -1)
            out3 = F.relu(self.FC3(out3))
            out_ls.append(out3)

            out4 = self.GAP4(features[k])

            k += 1

            out4 = out4.view(out4.size(0), -1)
            out4 = F.relu(self.FC4(out4))
            out_ls.append(out4)
        out = self.linear(torch.cat(out_ls, 1))
        return out


class LossNet_mnist(nn.Module):
    def __init__(
        self,
        feature_sizes=[84],
        num_channels=[84],
        interm_dim=128
    ):
        super(LossNet_mnist, self).__init__()
        
        # self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        # self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        # self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        # self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        # self.FC2 = nn.Linear(num_channels[1], interm_dim)
        # self.FC3 = nn.Linear(num_channels[2], interm_dim)
        # self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(interm_dim, 1)
    
    def forward(self, features):
        # out1 = self.GAP1(features[0])
        # out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(features[0]))

        # out2 = self.GAP2(features[1])
        # out2 = out2.view(out2.size(0), -1)
        # out2 = F.relu(self.FC2(out2))

        # out3 = self.GAP3(features[2])
        # out3 = out3.view(out3.size(0), -1)
        # out3 = F.relu(self.FC3(out3))

        # out4 = self.GAP4(features[3])
        # out4 = out4.view(out4.size(0), -1)
        # out4 = F.relu(self.FC4(out4))

        out = self.linear(out1)
        return out

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0)) # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    
    return loss


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4, encoder_out_dim=1024*2*2, first = True, last = True):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        if first == False:
            self.encoder = nn.Sequential(
                # nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
                # nn.BatchNorm2d(128),
                # nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                View((-1, 1024*2*2)),                                 # B, 1024*4*4
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                View((-1, 1024*2*2)),                                 # B, 1024*4*4
            )

        self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 1024*4*4),                           # B, 1024*8*8
            View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, r,x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z[0:r.shape[0]],r],1)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):  
        z = torch.cat([z[0:r.shape[0]], r], 1)
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch,
        epoch_loss, args):
    models['backbone'].train()
    models['module'].train()

    global iters
    for data in tqdm(dataloaders['train'], leave=False,
            total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[1].cuda()
            labels = data[2].cuda()

        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, _, features = models['backbone'].forward_with_features(inputs) 
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            for k in range(len(features)):
                features[k] = features[k].detach()
            # features[1] = features[1].detach()
            # features[2] = features[2].detach()
            # features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
        loss = m_backbone_loss + WEIGHT * m_module_loss 

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
    return loss


def train(models, method, criterion, optimizers, schedulers, dataloaders,
        num_epochs, epoch_loss, args):
    args.logger.info('>> Train a Model.')
    
    for epoch in tqdm(range(num_epochs)):

        loss = train_epoch(models, method, criterion, optimizers, dataloaders,
                epoch, epoch_loss, args)

        schedulers['backbone'].step()
        if method == 'lloss' or 'TA-VAAL':
            schedulers['module'].step()

    args.logger.info('>> Finished.')


def test(models, method, dataloader):
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for (_, inputs, labels) in dataloader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'].forward_with_features(inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for data in dataloader:
                yield data[1], data[2]
    else:
        while True:
            for _, img, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader,
        cycle, args):
    args.logger.info("Starting TA-VAAL training process")
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']
    
    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
        task_model = task_model.cuda()
        ranker = ranker.cuda()
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int((ADDENDUM*cycle+ SUBSET) * EPOCHV / args.batch_size)

    for iter_count in tqdm(range(train_iterations)):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()
        if iter_count == 0 :
            r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
            r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
        else:
            with torch.no_grad():
                _,_,features_l = task_model.forward_with_features(labeled_imgs)
                _,_,feature_u = task_model.forward_with_features(unlabeled_imgs)
                r_l = ranker(features_l)
                r_u = ranker(feature_u)
        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.sigmoid(r_l).detach()
            r_u_s = torch.sigmoid(r_u).detach()                 
        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            if labeled_imgs.shape[-1] < 32:
                padding_size = int((32 - labeled_imgs.shape[-1])/2)
                labeled_imgs = F.pad(labeled_imgs, (padding_size, padding_size, padding_size, padding_size))
                unlabeled_imgs = F.pad(unlabeled_imgs, (padding_size, padding_size, padding_size, padding_size))
            recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s,labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
            
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            if iter_count % 100 == 0:
                args.logger.info("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))


# Select the indices of the unlablled data according to the methods
def query_samples(model, train_set, subset, meta_set, cycle, args, nc=3, encoder_out_dim = 1024*2*2, z_dim = 32):
    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(
        train_set,
        batch_size=args.batch_size, 
        sampler=SubsetSequentialSampler(subset), 
        pin_memory=True,
    )
    labeled_loader = DataLoader(
        meta_set,
        batch_size=args.batch_size, 
        pin_memory=True,
    )
    vae = VAE(nc=nc, encoder_out_dim =encoder_out_dim, z_dim=z_dim, first=True, last=True)
    discriminator = Discriminator(32)
    
    models = {'backbone': model['backbone'], 'module': model['module'],'vae': vae, 'discriminator': discriminator}
    
    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
    optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

    train_vaal(models, optimizers, labeled_loader, unlabeled_loader, cycle+1,
            args)
    task_model = models['backbone']
    ranker = models['module']        
    all_preds, all_indices = [], []

    for indices, images, _ in unlabeled_loader:                       
        images = images.cuda()
        with torch.no_grad():
            _,_,features = task_model.forward_with_features(images)
            if images.shape[-1] < 32:
                padding_size = int((32 - images.shape[-1])/2)
                images = F.pad(images, (padding_size, padding_size, padding_size, padding_size))
            r = ranker(features)
            _, _, mu, _ = vae(torch.sigmoid(r),images)
            preds = discriminator(r,mu)

        preds = preds.cpu().data
        all_preds.extend(preds)
        all_indices.extend(indices)

    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1
    # select the points which the discriminator things are the most likely to be unlabeled
    _, arg = torch.sort(all_preds) 
    return arg


def main_train_taaval(args):
    method = 'TA-VAAL'
    args.logger.info("Dataset: %s"%args.dataset)
    args.logger.info("Method type:%s"%method)
    CYCLES = 2
    TRIALS = 1
    for trial in range(TRIALS):
        # Load training and testing dataset
        if args.dataset == "cifar100":
            NO_CLASSES = 100
        else:
            NO_CLASSES = 10
        train_set = torch.load(os.path.join(args.save_path, "cached_train_set"))

        adden = args.valid_count
        no_train = len(train_set)
        args.logger.info('The entire datasize is {}'.format(len(train_set)))       
        NUM_TRAIN = no_train
        ADDENDUM = adden
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        meta_set = torch.load(os.path.join(args.prev_save_path, "cached_meta_set"))
        valid_set = torch.load(os.path.join(args.prev_save_path, "cached_valid_set"))
        test_set = torch.load(os.path.join(args.prev_save_path, "cached_test_set"))

        unlabeled_set = list(range(len(train_set)))
        # meta_loader = DataLoader(
        #     meta_set,
        #     batch_size=args.batch_size, 
        #     pin_memory=True,
        #     # drop_last=True,
        # )

        train_sampler = DistributedSampler(
            train_set,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle = False
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            # num_workers=4*4, #args.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=train_sampler,
        )

        meta_sampler = RandomSampler(meta_set, replacement=True, num_samples=args.epochs*len(train_loader)*args.batch_size*10)
        meta_loader = DataLoader(
            meta_set,
            batch_size=args.test_batch_size,
            num_workers=0,#args.num_workers,
            pin_memory=True,
            sampler=meta_sampler,
        )

        # train_loader = DataLoader(
        #     train_set,
        #     batch_size=args.batch_size, 
        #     pin_memory=True,
            
        #     # drop_last=True,
        # )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size, 
            pin_memory=True,
            # drop_last=True,
        )
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
        dataloaders = {'meta': meta_loader, 'valid': valid_loader, 'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):
            
            args.logger.info("Meta set size: {}".format(len(meta_set)))
            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                resnet = resnet34(num_classes=NO_CLASSES).cuda()
                loss_module = LossNet().cuda()
                if args.dataset == 'retina':
                    resnet = resnet34_imagenet(pretrained=True, first=args.biased_flip, last=True)
                    resnet.fc = nn.Linear(512, 2)
                    resnet = resnet.cuda()
                    loss_module = LossNet(first=args.biased_flip, last=True, num_channels=[64, 128, 1024, 2048]).cuda()
                
                if args.dataset == 'imagenet':
                    resnet = resnet34_imagenet(pretrained=True, first=True, last=True)
                    resnet.fc = nn.Linear(512, 10)
                    resnet = resnet.cuda()
                    loss_module = LossNet(first=True, last=True).cuda()

                nc = 3
                encoder_out_dim=1024*2*2
                z_dim=32

                if args.dataset == 'MNIST':
                    # if args.model_type == 'lenet':
                    resnet = LeNet5().cuda()
                    loss_module = LossNet_mnist().cuda()
                    nc=1
                    encoder_out_dim=1024
                    z_dim=28
                    # else:
                    #     resnet = DNN_three_layers(args.nce_k, low_dim=args.low_dim).cuda()
                    
            if args.use_pretrained_model:
                resnet = main.main_train.load_checkpoint2(args, resnet)

            models = {'backbone': resnet, 'module': loss_module}
            torch.backends.cudnn.benchmark = True
            
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            if args.dataset == 'MNIST':
                optim_backbone = optim.SGD(
                    models['backbone'].parameters(),
                    lr=args.lr, 
                    momentum=0.9,
                    weight_decay=5e-4,
                )
            else:
                optim_backbone = torch.optim.SGD(resnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                optim_backbone.param_groups[0]['initial_lr'] = args.lr
 
            sched_backbone = lr_scheduler.MultiStepLR(
                optim_backbone,
                milestones=MILESTONES,
            )
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            optim_module   = optim.SGD(
                models['module'].parameters(),
                lr=args.lr, 
                momentum=0.9,
                weight_decay=5e-4,
            )
            sched_module   = lr_scheduler.MultiStepLR(
                optim_module,
                milestones=MILESTONES,
            )
            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
            
            # Training and testing
            if cycle > 0:
                prev_weights = None
                main.main_train.meta_learning_model(
                    args,
                    args.logger,
                    DDP(models['backbone'], device_ids=[args.local_rank]),
                    optimizers['backbone'],
                    criterion,
                    criterion,
                    dataloaders['train'],
                    dataloaders['meta'],
                    dataloaders['valid'],
                    dataloaders['test'],
                    mnist_to_device,
                    scheduler=schedulers['backbone'],
                    cached_w_array=prev_weights,
                    target_id=None,
                    start_ep=0,
                    mile_stones_epochs=MILESTONES,
                    heuristic=None,
                    gt_training_labels=None,
                )
            acc = test(models, method, dataloaders['test'])
            args.logger.info(
                'Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                    trial+1, TRIALS, cycle+1, CYCLES, len(meta_set), acc))
            # np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                args.logger.info("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(
                models,
                train_set,
                subset, 
                meta_set,
                cycle,
                args,
                nc = nc,
                encoder_out_dim = encoder_out_dim,
                # z_dim = z_dim
            )

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            meta_set_ids = list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            args.logger.info("Number of selected samples: {}".format(len(meta_set_ids)))

            # Create a new dataloader for the updated labeled dataset
            origin_labels = torch.load(os.path.join(args.save_path, "cached_train_origin_labels"))
            train_set, new_meta_set = split_train_valid_set_by_ids(args, train_set,
                    origin_labels, meta_set_ids, unlabeled_set)
            meta_set = meta_set.concat_validset(meta_set, new_meta_set)

            torch.save(meta_set, os.path.join(args.save_path, "cached_meta_set"))
            torch.save(train_set, os.path.join(args.save_path, "cached_train_set"))

            meta_sampler = RandomSampler(meta_set, replacement=True, num_samples=args.epochs*len(train_loader)*args.batch_size*10)
            meta_loader = DataLoader(
                meta_set,
                batch_size=args.test_batch_size,
                num_workers=0,#args.num_workers,
                pin_memory=True,
                sampler=meta_sampler,
            )



            dataloaders['meta'] = meta_loader
            # DataLoader(
            #     meta_set,
            #     batch_size=args.batch_size, 
            #     pin_memory=True,
            # )

            train_sampler = DistributedSampler(
                train_set,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle = False
            )

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                # num_workers=4*4, #args.num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=train_sampler,
            )
            dataloaders['train'] = train_loader
            # DataLoader(
            #     train_set,
            #     batch_size=args.batch_size, 
            #     pin_memory=True,
            # )

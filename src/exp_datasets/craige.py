import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import numpy as np
import math
import apricot
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import random

class Strategy:
    def __init__(self,X, Y, unlabeled_x, transform, net, handler, nclasses, args={}): #
        
        self.X = X
        self.Y = Y
        self.unlabeled_x = unlabeled_x
        self.model = net
        self.handler = handler
        self.transform = transform
        self.target_classes = nclasses
        self.args = args
        if 'batch_size' not in args:
            args['batch_size'] = 1
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def select(self, budget):
        pass

    def update_data(self,X,Y,unlabeled_x): #
        self.X = X
        self.Y = Y
        self.unlabeled_x = unlabeled_x

    def update_model(self, clf):
        self.model = clf

    def save_state(self, filename):


        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_state(self, filename):

        with open(filename, 'rb') as f:
            self = pickle.load(f)

    def predict(self,X, useloader=True):
    
        self.model.eval()
        P = torch.zeros(X.shape[0]).long()
        with torch.no_grad():

            if useloader:
                loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])
                for idxs, x in loader_te:
                    x = x.to(self.device)  
                    out = self.model(x)
                    pred = out.max(1)[1]
                    P[idxs] = pred.data.cpu()
            else:            
                x=X
                out = self.model(x)
                pred = out.max(1)[1]
                P = pred.data.cpu()

        return P

    def predict_prob(self,X):

        loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])
        self.model.eval()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for idxs, x in loader_te:
                x = x.to(self.device)                  
                out = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self,X, n_drop):

        loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])
        self.model.train()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                # print('n_drop {}/{}'.format(i+1, n_drop))
                for idxs, x in loader_te:

                    x = x.to(self.device)   
                    out = self.model(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self,X, n_drop):
        
        loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])
        self.model.train()
        probs = torch.zeros([n_drop, X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                # print('n_drop {}/{}'.format(i+1, n_drop))
                for idxs, x in loader_te:
                    x = x.to(self.device)
                    out = self.model(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self,X):
        
        loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])
        self.model.eval()
        embedding = None#torch.zeros([X.shape[0], self.model.get_embedding_dim()])

        with torch.no_grad():
            for idxs, x in loader_te:
                x = x.to(self.device)  
                # out= self.model(x)
                e1 = self.model.feature_forward(x)
                if embedding is None:
                    embedding = torch.zeros([X.shape[0], e1.shape[1]])
                embedding[idxs] = e1.data.cpu()
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self,X,Y=None, bias_grad=True):
        
        # embDim = self.model.get_embedding_dim()
        
        nLab = self.target_classes

        embedding = None

                
        loader_te = DataLoader(self.handler(X, self.transform),shuffle=False, batch_size = self.args['batch_size'])

        with torch.no_grad():
            for idxs, x in loader_te:
                x = x.to(self.device)
                # out, l1 = self.model(x,last=True)
                l1 = self.model.feature_forward(x)
                if embedding is None:
                    embedding = torch.zeros([x.shape[0], l1.shape[1]],device=self.device)
                # if bias_grad:
                #     embedding = torch.zeros([x.shape[0], (embDim+1)*nLab],device=self.device)
                # else:
                #     embedding = torch.zeros([x.shape[0], embDim * nLab],device=self.device)
                out = self.model(x)
                data = F.softmax(out, dim=1)

                outputs = torch.zeros(x.shape[0], nLab).to(self.device)
                if Y is None:
                    y_trn = self.predict(x, useloader=False)
                else:
                    y_trn = torch.tensor(Y[idxs])
                y_trn = y_trn.to(self.device).long()
                outputs.scatter_(1, y_trn.view(-1, 1), 1)
                l0_grads = data - outputs
                l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                l1_grads = l0_expand * l1.repeat(1, nLab)
                
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                
                if bias_grad:
                    embedding[idxs] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    embedding[idxs] = l1_grads

        return embedding

class DataSelectionStrategy(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        valloader: class
            Loading the validation data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
        linear_layer: bool
            If True, we use the last fc layer weights and biases gradients
            If False, we use the last fc layer biases gradients
        loss: class
            PyTorch Loss function
    """

    def __init__(self, trainloader, valloader, model, num_classes, linear_layer, loss, device):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device

    def select(self, budget, model_params):
        pass

    def get_labels(self, valid=False):
        for batch_idx, (_, inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                self.trn_lbls = targets.view(-1, 1)
            else:
                self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            for batch_idx, (_, inputs, targets) in enumerate(self.valloader):
                if batch_idx == 0:
                    self.val_lbls = targets.view(-1, 1)
                else:
                    self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)

    def compute_gradients(self, valid=False, batch=False, perClass=False):
        """
        Computes the gradient of each element.
        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.
        Using different loss functions, the way we calculate the gradients will change.
        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:
        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,
        where :math:`N` is the batch size.
        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:
        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,
        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        batch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if perClass:
            # embDim = self.model.get_embedding_dim()
            for batch_idx, (_, inputs, targets) in enumerate(self.pctrainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    # out, l1 = self.model(inputs, last=True, freeze=True)
                    out = self.model(inputs)
                    l1 = self.model.feature_forward(inputs)
                    embDim = l1.shape[1]
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out = self.model(inputs)
                    l1 = self.model.feature_forward(inputs)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.grads_per_elem = l0_grads

            if valid:
                for batch_idx, (_, inputs, targets) in enumerate(self.pcvalloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if batch_idx == 0:
                        l1 = self.model.feature_forward(inputs)
                        out = self.model(inputs)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        # out, l1 = self.model(inputs, last=True, freeze=True)
                        l1 = self.model.feature_forward(inputs)
                        out = self.model(inputs)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                torch.cuda.empty_cache()
                if self.linear_layer:
                    self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    self.val_grads_per_elem = l0_grads
        else:
            # embDim = self.model.get_embedding_dim()
            for batch_idx, (_, inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out = self.model(inputs)
                    l1 = self.model.feature_forward(inputs)
                    embDim = l1.shape[1]
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if batch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

            torch.cuda.empty_cache()

            if self.linear_layer:
                self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.grads_per_elem = l0_grads
            if valid:
                for batch_idx, (_, inputs, targets) in enumerate(self.valloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if batch_idx == 0:
                        # out, l1 = self.model(inputs, last=True, freeze=True)
                        l1 = self.model.feature_forward(inputs)
                        out = self.model(inputs)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        # out, l1 = self.model(inputs, last=True, freeze=True)
                        l1 = self.model.feature_forward(inputs)
                        out = self.model(inputs)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                        if batch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                torch.cuda.empty_cache()
                if self.linear_layer:
                    self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    self.val_grads_per_elem = l0_grads

    def update_model(self, model_params):
        """
        Update the models parameters
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)

def calculate_class_budgets(budget, num_classes, trn_lbls, N_trn):
    # Tabulate class populations
    class_pops = list()
    for i in range(num_classes):
        trn_subset_idx = torch.where(trn_lbls == i)[0].tolist()
        class_pops.append(len(trn_subset_idx))

    # Assign initial class budgets, where each class gets at least 1 element (if there are any elements to choose)
    class_budgets = list()
    class_not_zero = list()
    for i in range(num_classes):
        if class_pops[i] > 0:
            class_budgets.append(1)
            class_not_zero.append(i)
        else:
            class_budgets.append(0)

    # Check if we have violated the budget. If so, pick random indices to set to 0.
    current_class_budget_total = 0
    for class_budget in class_budgets:
        current_class_budget_total = current_class_budget_total + class_budget

    if current_class_budget_total > budget:
        set_zero_indices = random.sample(class_not_zero, current_class_budget_total - budget)
			
        for i in set_zero_indices:
            class_budgets[i] = 0
		
    # We can proceed to adjust class budgets if we have not met the budget.
    # Note: if these two quantities are equal, we do not need to do anything.
    elif current_class_budget_total < budget:		
        # Calculate the remaining budget
        remaining_budget = budget - current_class_budget_total

        # Calculate fractions
        floored_class_budgets = list()
        for i in range(num_classes):
            # Fraction is computed off remaining budget to add. Class population is adjusted to remove freebee element (if present).
            # Total elements in consideration needs to remove already guaranteed elements (current_class_budget_total).
            # Add previous freebee element to remaining fractions
            class_budget = class_budgets[i] + remaining_budget * (class_pops[i] - class_budgets[i]) / (N_trn - current_class_budget_total)
            floored_class_budgets.append((i, math.floor(class_budget), class_budget - math.floor(class_budget)))

        # Sort the budgets to partition remaining budget in descending order of floor error.
        list.sort(floored_class_budgets, key=lambda x: x[2], reverse=True)

        # Calculate floored budget sum
        floored_sum = 0
        for _, floored_class_budget, _ in floored_class_budgets:
            floored_sum = floored_sum + floored_class_budget

        # Calculate new remaining total budget
        remaining_budget = budget - floored_sum
        index_iter = 0

        while remaining_budget > 0:
            class_index, class_budget, class_floor_err = floored_class_budgets[index_iter]
            class_budget = class_budget + 1
            floored_class_budgets[index_iter] = (class_index, class_budget, class_floor_err)

            index_iter = index_iter + 1
            remaining_budget = remaining_budget - 1	

        # Rearrange budgets to be sorted by class
        list.sort(floored_class_budgets, key=lambda x: x[0])

        # Override class budgets list with new values
        for i in range(num_classes):
            class_budgets[i] = floored_class_budgets[i][1]
                
        return class_budgets

class CRAIGStrategy(DataSelectionStrategy):
    """
    Implementation of CRAIG Strategy from the paper :footcite:`mirzasoleiman2020coresets` for supervised learning frameworks.
    CRAIG strategy tries to solve the optimization problem given below for convex loss functions:
    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| x^i - x^j \\|
    In the above equation, :math:`\\mathcal{U}` denotes the training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.
    Since, the above optimization problem is not dependent on model parameters, we run the subset selection only once right before the start of the training.
    CRAIG strategy tries to solve the optimization problem given below for non-convex loss functions:
    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| \\nabla_{\\theta} {L_T}^i(\\theta) - \\nabla_{\\theta} {L_T}^j(\\theta) \\|
    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round,
    and :math:`k` is the budget for the subset. In this case, CRAIG acts an adaptive subset selection strategy that selects a new subset every epoch.
    Both the optimization problems given above are an instance of facility location problems which is a submodular function. Hence, it can be optimally solved using greedy selection methods.
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        Type of selection:
         - 'PerClass': PerClass Implementation where the facility location problem is solved for each class seperately for speed ups.
         - 'Supervised':  Supervised Implementation where the facility location problem is solved using a sparse similarity matrix by assigning the similarity of a point with other points of different class to zero.
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex, selection_type, optimizer='lazy'):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.optimizer = optimizer

    def distance(self, x, y, exp=2):
        """
        Compute the distance.
        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)
        Returns
        ----------
        dist: Tensor
            Output tensor
        """

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        # dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        """
        Compute the score of the indices.
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """

        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                    sampler=SubsetRandomSampler(idxs),
                                                    pin_memory=True)
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        if self.if_convex:
            for batch_idx, (_,inputs, targets) in enumerate(subset_loader):
                inputs, targets = inputs, targets
                if self.selection_type == 'PerBatch':
                    self.N += 1
                    g_is.append(inputs.view(inputs.size()[0], -1).mean(dim=0).view(1, -1))
                else:
                    self.N += inputs.size()[0]
                    g_is.append(inputs.view(inputs.size()[0], -1))
        else:
            # embDim = self.model.get_embedding_dim()
            for batch_idx, (_, inputs, targets) in enumerate(subset_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if self.selection_type == 'PerBatch':
                    self.N += 1
                else:
                    self.N += inputs.size()[0]
                # out, l1 = self.model(inputs, freeze=True, last=True)
                out = self.model(inputs)
                l1 = self.model.feature_forward(inputs)
                embDim = l1.shape[1]
                loss = self.loss(out, targets).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if self.selection_type == 'PerBatch':
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(1, -1).detach().cpu())
                    else:
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1).detach().cpu())
                else:
                    if self.selection_type == 'PerBatch':
                        g_is.append(l0_grads.mean(dim=0).view(1, -1).detach().cpu())
                    else:
                        g_is.append(l0_grads.detach().cpu())

        self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
        first_i = True
        if self.selection_type == 'PerBatch':
            g_is = torch.cat(g_is, dim=0)
            self.dist_mat = self.distance(g_is, g_is).cpu()
        else:
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.
        Parameters
        ----------
        idxs: list
            The indices
        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        if self.selection_type in ['PerClass', 'PerBatch']:
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in rep:
                gamma[i] += 1
        elif self.selection_type == 'Supervised':
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in range(rep.shape[1]):
                gamma[rep[0, i]] += 1
        return gamma

    def get_similarity_kernel(self):
        """
        Obtain the similarity kernel.
        Returns
        ----------
        kernel: ndarray
            Array of kernel values
        """
        for batch_idx, (_, inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        kernel = np.zeros((labels.shape[0], labels.shape[0]))
        for target in np.unique(labels):
            x = np.where(labels == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def select(self, budget, model_params):
        """
        Data selection method using different submodular optimization
        functions.
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'
        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """

        for batch_idx, (_, inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        # per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'PerClass':
            self.get_labels(False)
            class_budgets = calculate_class_budgets(budget, self.num_classes, self.trn_lbls, self.N_trn)
            
            for i in range(self.num_classes):
                idxs = torch.where(labels == i)[0]
                self.compute_score(model_params, idxs)
                fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=class_budgets[i],
                                                                                  optimizer=self.optimizer)
                sim_sub = fl.fit_transform(self.dist_mat)
                greedyList = list(np.argmax(sim_sub, axis=1))
                gamma = self.compute_gamma(greedyList)
                total_greedy_list.extend(idxs[greedyList])
                gammas.extend(gamma)
            rand_indices = np.random.permutation(len(total_greedy_list))
            total_greedy_list = list(np.array(total_greedy_list)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])
        elif self.selection_type == 'Supervised':
            for i in range(self.num_classes):
                if i == 0:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = idxs.repeat_interleave(N)
                    col = idxs.repeat(N)
                    data = self.dist_mat.flatten()
                else:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    if N <= 0:
                        continue
                    self.compute_score(model_params, idxs)
                    row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                    col = torch.cat((col, idxs.repeat(N)), dim=0)
                    data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
            sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
            self.dist_mat = sparse_simmat
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget, optimizer=self.optimizer)
            sim_sub = fl.fit_transform(sparse_simmat)
            total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas = self.compute_gamma(total_greedy_list)
        elif self.selection_type == 'PerBatch':
            idxs = torch.arange(self.N_trn)
            N = len(idxs)
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=math.ceil(
                                                                                  budget / self.trainloader.batch_size),
                                                                              optimizer=self.optimizer)
            sim_sub = fl.fit_transform(self.dist_mat)
            temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas_temp = self.compute_gamma(temp_list)
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(temp_list)):
                tmp = batch_wise_indices[temp_list[i]]
                total_greedy_list.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        return total_greedy_list, gammas


class SupervisedSelectHandler(Dataset):
    
    def __init__(self, wrapped_handler):
        
        self.wrapped_handler = wrapped_handler
        
    def __getitem__(self, index):
        
        if self.wrapped_handler.select == False:
            
            x, y, index = self.wrapped_handler.__getitem__(index)
            return x, y
        
        else:
        
            x, index = self.wrapped_handler.__getitem__(index)
            return x
        
    def __len__(self):
        
        return len(self.wrapped_handler.X)

class CRAIGActive(Strategy):
    
    """
    This is an implementation of an active learning variant of CRAIG from the paper Coresets for Data-efficient 
    Training of Machine Learning Models :footcite:`Mirzasoleiman2020craig`. This algorithm calculates hypothesized 
    labels for each of the unlabeled points and feeds this hypothesized set to the original CRAIG algorithm. The 
    selected points from CRAIG are used as the queried points for this algorithm.

    
    Parameters
    ----------
    X: Numpy array 
        Features of the labled set of points 
    Y: Numpy array
        Lables of the labled set of points 
    unlabeled_x: Numpy array
        Features of the unlabled set of points 
    net: class object
        Model architecture used for training. Could be instance of models defined in `distil.utils.models` or something similar.
    criterion: class object
        The loss type used in training. Could be instance of torch.nn.* losses or torch functionals.
    handler: class object
        It should be a subclass of torch.utils.data.Dataset i.e, have __getitem__ and __len__ methods implemented, so that is could be passed to pytorch DataLoader.Could be instance of handlers defined in `distil.utils.DataHandler` or something similar.
    nclasses: int 
        No. of classes in tha dataset
    lrn_rate: float
        The learning rate used in training. Used by the CRAIG algorithm.
    selection_type: string
        Should be one of "PerClass", "Supervised", or "PerBatch". Selects which approximation method is used.
    linear_layer: bool
        Sets whether to include the last linear layer parameters as part of the gradient computation.
    args: dictionary
        This dictionary should have keys 'batch_size' and  'lr'. 
        'lr' should be the learning rate used for training. 'batch_size'  'batch_size' should be such 
        that one can exploit the benefits of tensorization while honouring the resourse constraits.
    """
    
    def __init__(self, X, Y, unlabeled_x, net, criterion, handler, handler2, nclasses, lrn_rate, selection_type, linear_layer, transform, args={}):
        
        # Run super constructor
        super(CRAIGActive, self).__init__(X, Y, unlabeled_x, transform, net, handler, nclasses, args)
        self.criterion = criterion
        self.transform = transform
        self.lrn_rate = lrn_rate
        self.selection_type = selection_type
        self.linear_layer = linear_layer
        self.handler2 = handler2

    def select(self, budget):
        
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        subset_idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        
        # Compute hypothesize labels using model
        hypothesized_labels = self.predict(self.unlabeled_x)
        
        # Create a DataLoader from hypothesized labels and unlabeled points that will work with CORDS
        # cords_handler = SupervisedSelectHandler(self.handler2(self.unlabeled_x, hypothesized_labels.numpy(), self.transform))
        cords_handler = self.handler2(self.unlabeled_x, hypothesized_labels.numpy(), self.transform)

        trainloader = DataLoader(cords_handler, shuffle=False, batch_size = self.args['batch_size'])

        # Match on the hypothesized labels
        validloader = trainloader
        
        # Perform selection
        cached_state_dict = copy.deepcopy(self.model.state_dict())
        clone_dict = copy.deepcopy(self.model.state_dict())

        # Create CORDS CRAIGStrategy to select datapoints
        setf_model = CRAIGStrategy(trainloader, validloader, self.model, self.criterion, self.device, self.target_classes, self.linear_layer, False, self.selection_type)
        subset_idxs, _ = setf_model.select(budget, clone_dict)
        self.model.load_state_dict(cached_state_dict)
        return subset_idxs
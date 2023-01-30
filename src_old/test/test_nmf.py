import torch


data = torch.rand((100,10))


x = torch.rand((1,10), requires_grad=True)



for k in range(100):

    loss =  1/torch.sum(torch.abs(torch.mm(data, torch.t(x)))**2)

    # loss.backward()
    x_grad = torch.autograd.grad(loss,x)[0]

    x = x - 0.5*x_grad

    # x.grad.zero_()

    print(torch.norm(x_grad))
    # print(x)


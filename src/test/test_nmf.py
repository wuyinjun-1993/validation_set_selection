import torch


def do_nmf(V, rdim, nmf_iter = 5000, lamb = 0.0001):
    V = V/torch.max(V)
    # rng(1997)
    V_shape = list(V.shape)
    W = torch.abs(torch.randn(V_shape[0],rdim)) 
    H = torch.abs(torch.randn(rdim,V_shape[1]))

    itr = 0
    
    while itr < nmf_iter:
        itr = itr + 1
        
        # % resecale columns of W such that they are unit norm
        # W = W/torch.mm(torch.ones(V_shape[0],1), torch.sqrt(torch.sum(W**2,1)).view(1,-1))
        W = W/(torch.norm(W, dim = 0).view(1,-1))
        
        # % reconstruction of V
        R = torch.mm(W, H)# W*H
        
        # % update sparse activations
        H = H*((torch.mm(torch.t(W), V))/(torch.mm(torch.t(W),R) + lamb))
        
        # % recompute reconstruction
        R = torch.mm(W, H)

        W = W*torch.mm(V, torch.t(H))/(torch.mm(R, torch.t(H)) + lamb)

        # % update W (non-parametrically)
        # for j=1:size(W,2) # % columns 
#         for j in range(W.shape[1]):
# # %             % verbatim algo described in paper, for verification of
# # %             % optimized 
# # %             num = zeros(size(V,1),1); den = zeros(size(V,1),1);
# # %             for i=1:size(V,1)
# # %                num = num + H(j,i)*(V(i,:)' + (R(i,:)*W(:,j))*W(:,j));
# # %                den = den + H(j,i)*(R(i,:)' + (V(i,:)*W(:,j))*W(:,j));
# # %             end
            
#             num = torch.mm((torch.t(V) + torch.t(torch.mm((R*W[:,j].view(-1,1)),torch.ones([1,V_shape[0]])))*W[:,j]), torch.t(H[j,:]))
#             den = torch.mm((torch.t(R) + torch.t(torch.mm((V*W[:,j].view(-1,1)),torch.ones([1,V_shape[0]])))*W[:,j]), torch.t(H[j,:]))
#             W[:,j] = W[:,j]*(num/den)
        err = torch.norm(V - torch.mm(W,H))
        print("error::", itr, err.item())

    print()

    # _,pp =torch.sort(sum(W**2,1),descending=True)
    # W=W[:,pp]
    # H=H[pp,:]
    # newerr = torch.norm(abs(V-torch.mm(W,H)))
    # print("new error::", newerr)


V = torch.rand([100,50])
do_nmf(V, 10)


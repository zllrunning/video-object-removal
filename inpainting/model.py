import torch
from torch import nn
from inpainting.models import vinet
import pdb

def generate_model(opt):

    try: 
        assert(opt.model == 'vinet_final')
        model = vinet.VINet_final(opt=opt)
    except:
        print('Model name should be: vinet_final')

    assert(opt.no_cuda is False)
    model = model.cuda()
    model = nn.DataParallel(model)
    loaded, empty = 0,0
    if opt.pretrain_path:
        print('Loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)

        child_dict = model.state_dict()
        parent_list = pretrain['state_dict'].keys()
        parent_dict = {}
        for chi,_ in child_dict.items():
            if chi in parent_list:
                parent_dict[chi] = pretrain['state_dict'][chi]
                #print('Loaded: ',chi)
                loaded += 1
            else:
                #print('Empty:',chi)
                empty += 1
        print('Loaded: %d/%d params'%(loaded, loaded+empty))
        child_dict.update(parent_dict)
        model.load_state_dict(child_dict)   

    return model, model.parameters()

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
sys.path.append('/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/code')

from common import *
from datasets.data   import *

# from HED.hednet import HEDResnet34 as Net
# from HED.hednet import HyperAttResnet34 as Net
# from HED.hednet import HedSENet50 as Net
# from HED.hednet import HedSEResXt101 as Net
from models.hypercolumn_unets import HedSEResXt154 as Net


def load_pretrain_file(net, pretrain_file, skip=[]):

    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip):
            continue

        state_dict[key] = pretrain_state_dict[key]
    net.load_state_dict(state_dict)
    return net


def train_augment(image,mask,index):
        cache = Struct(image = image.copy(), mask = mask.copy())

        if np.random.rand() < 0.5:
             image, mask = do_horizontal_flip2(image, mask)
             pass

        if np.random.rand() < 0.5:
            c = np.random.choice(4)
            if c==0:
                image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)
            if c==1:
                image, mask = do_elastic_transform2(image, mask, grid=10,
                                                distort=np.random.uniform(0,0.1))
            if c==2:
                image, mask = do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1,
                                                  angle=np.random.uniform(0,10))
            
            if c == 3:
                image, mask = do_horizontal_shear2(image, mask, np.random.uniform(-0.1, 0.1))
                pass

        if np.random.rand() < 0.5:
            c = np.random.choice(4)
            if c==0:
                image = do_brightness_shift(image,np.random.uniform(-0.05,+0.05))
            if c==1:
                image = do_brightness_multiply(image,np.random.uniform(1-0.05,1+0.05))
            if c==2:
                image = do_gamma(image,np.random.uniform(1-0.05,1+0.05))
            if c==3:
                image = do_invert_intensity(image)


        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
        # image, mask = do_resize2(image, mask, 128, 128)
        return image,mask,index,cache

def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_center_pad_to_factor2(image, mask, factor=32)
    # image, mask = do_resize2(image, mask, 128, 128)
    return image,mask,index, cache

#-------------------------------------------
### training                ##############################################################

def do_valid( net, valid_loader ):

    valid_num  = 0
    valid_loss = np.zeros(3,np.float32)

    predicts = []
    truths   = []

    for input, truth, index, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = net(input)
            prob  = F.sigmoid(logit)
            loss  = net.criterion(logit, truth)
            dice  = net.metric(logit, truth)

        batch_size = len(index)
        valid_loss += batch_size*np.array(( loss.item(), dice.item(), 0))
        valid_num += batch_size

        predicts.append(prob.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))
    valid_loss  = valid_loss/valid_num

    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths   = np.concatenate(truths).squeeze()
    predicts = predicts[:,DY0:DY0+IMAGE_HEIGHT, DX0:DX0+IMAGE_WIDTH]
    truths   = truths  [:,DY0:DY0+IMAGE_HEIGHT, DX0:DX0+IMAGE_WIDTH]
    # predicts = resize_masks(predicts, 101, 101)
    # truths   = resize_masks(truths, 101, 101)

    precision, result, threshold  = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss


def run_train():

    out_dir = '/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/HedSENeXt154_128/fold5'
    initial_checkpoint = out_dir + '/checkpoint/best_model_20180926.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/hedresnet34_bn_scSE_resize128/checkpoint/best_model_20180904.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet34_resize128/checkpoint/best_model.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet50_bn_resize128/checkpoint/best_model.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet34_resize128/checkpoint/best_model.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet50_bn_resize128/checkpoint/00006000_model.pth'
        #  
         
        
    pretrain_file = None#'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/HedSENeXt101_128/fold6/checkpoint/best_model_20180926.pth'
        # '/data/jiuchao/resnet34-333f7ec4.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet50_bn_256/checkpoint/00021300_model.pth'
        # '/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet34_bn_256/checkpoint/00029900_model.pth'
        #'/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet34_256/checkpoint/00029500_model.pth'
        #'/data/jiuchao/resnet50-19c8e357.pth' #'/data/jiuchao/resnet34-333f7ec4.pth' \
        # '/data/jiuchao/resnet50-19c8e357.pth' #
                


    # setup --------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 16

    train_dataset = TsgDataset('list_train5_3600', train_augment, 'train')
    train_loader  = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    valid_dataset = TsgDataset('list_valid5_400', valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset.split = %s\n'%(train_dataset.split))
    log.write('valid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\n')

    ## net ------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        # load_pretrain_file(net, pretrain_file)
        net.load_pretrain(pretrain_file)

    
    log.write('%s\n'%(type(net)))
    log.write('\n')

    ## optimizer ----------------------------------
    num_iters   = 225 * 500
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100

    schduler = CyclicScheduler(min_lr=0.0005, max_lr=0.005, step=10725) #StepScheduler([(0, 0.01), (1000, 0.005), (3000, 0.001)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch = 0
    best_metric = 0.0

    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('model','optimizer'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']

        # rate = get_learning_rate(optimizer)  #load all except learning rate
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # adjust_learning_rate(optimizer, rate)
        pass

    print("best metric: %0.3f"%best_metric)
    ## start training here! ##############################################
    log.write('** start training here! **\n')

    #log.write(' samples_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate    iter   epoch   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0
    iter = 0
    i    = 0

    start = timer()

    while  iter<num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        optimizer.zero_grad()
        for input, truth, index, cache in train_loader:
            
            len_train_dataset = len(train_dataset)
            batch_size = len(index)
            iter = i + start_iter
            epoch = (iter-start_iter)*batch_size/len_train_dataset + start_epoch
            num_samples = epoch*len_train_dataset


            if iter % iter_valid==0:
                net.set_mode('valid')
                valid_loss = do_valid(net, valid_loader)
                net.set_mode('train')

                #save model
                if(valid_loss[2] >= best_metric):
                    best_metric = valid_loss[2]
                    torch.save(net.state_dict(),out_dir +'/checkpoint/best_model_20180926.pth')
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                        'best_metric': best_metric,
                    }, out_dir +'/checkpoint/best_optimizer_20180926.pth')

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str(timer() - start)))
                time.sleep(0.01)

            # if iter % 11250 == 0:
            #     schduler._reset(new_min_lr=0.001 / (snapshot // 4 + 1), new_max_lr=0.01 * (snapshot // 4 + 1))
            #     snapshot += 1
            #     best_metric = 0.0


            # learning rate schduler -------------
            if schduler is not None:
                lr = schduler.get_rate(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            net.set_mode('train')

            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net,input) #net(input)
            loss  = net.criterion(logit, truth)
            dice  = net.metric(logit, truth)


            loss.backward()
            # log.write("\n################################### New  Epoch #############################", is_terminal=0)
            # for name, param in net.named_parameters():
            #     if param.requires_grad:
            #         if np.sum(param.grad.data.cpu().numpy()) == 0.0:
            #             log.write('\n' + name, is_terminal=0)
            optimizer.step()
            optimizer.zero_grad()

            # print statistics  ------------
            batch_loss = np.array((
                           loss.item(),
                           dice.item(),
                           0, 0, 0, 0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s ' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str(timer() - start)), end='',flush=True)
            i=i+1

        pass    #-- end of one data loader --
    pass    #-- end of all iterations --

    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/last_model_20180926.pth')
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
            'best_metric': best_metric,
        }, out_dir +'/checkpoint/last_optimizer_20180926.pth')

    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')

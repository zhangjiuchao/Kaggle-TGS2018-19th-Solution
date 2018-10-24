import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
sys.path.append('/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/code')

from common import *
from data import *

# from unet_resnet.model_resnet import UNetResnet34 as Net
# from unet_resnet.model_resnet_bn import UNetResNet34 as Net
# from unet_resnet.model_resnet_bn import UNetResNet50 as Net
# from HED.hednet import HEDResnet34 as Net
# from HED.hednet import HyperAttResnet34 as Net
# from HED.hednet import HedDensenet121 as Net
# from HED.hednet import HedSENet50 as Net
# from HED.hednet import DeepHedSENeXt50 as Net
# from HED.hednet import HesSEResXt101 as Net
from HED.hednet import HedSEResXt154 as Net

out_dir = '/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/HedSENeXt154_128/fold5'

initial_checkpoint = out_dir + '/checkpoint/best_model_20180926.pth'

# split, mode = 'list_valid7_400', 'valid'
split, mode = 'list_test_18000', 'test'


### submitting   ##############################################################

#augment == 'flip'
def test_augment_flip(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if mask==[]:
        image = do_horizontal_flip(image)
        image = do_center_pad_to_factor(image, factor=32)
        # image = do_resize(image, 128, 128)
    else:
        image, mask = do_horizontal_flip2(image, mask)
        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
        # image, mask = do_resize2(image, mask, 128, 128)

    return image, mask, index, cache



def test_unaugment_flip(prob):
    dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    # prob = resize_masks(prob, 101, 101)
    prob = prob[:,:,::-1]
    return prob

#---------------------
#augment == 'null' :
def test_augment_null(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if mask==[]:
        image = do_center_pad_to_factor(image, factor=32)
        # image = do_resize(image, 128, 128)
    else:
        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
        # image, mask = do_resize2(image, mask, 128, 128)

    return image, mask, index, cache


def test_unaugment_null(prob):
    dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    # prob = resize_masks(prob, 101, 101)
    return prob





##############################################################################################

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
    precision, result, threshold  = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss


def run_predict(augment):

    if augment == 'null':
        test_augment   = test_augment_null
        test_unaugment = test_unaugment_null
    if augment == 'flip':
        test_augment   = test_augment_flip
        test_unaugment = test_unaugment_flip
    #....................................................


    ## setup  -----------------
    os.makedirs(out_dir +'/test/' + split, exist_ok=True)
    # os.makedirs(out_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.test.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 32

    test_dataset = TsgDataset(split, test_augment, mode)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(test_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('\n')



    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        checkpoint  = torch.load(initial_checkpoint.replace('model','optimizer'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        log.write('best_metric: {}\n'.format(best_metric))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    # net.set_mode('test')
    # valid_loss = do_valid(net, test_loader)
    # print(valid_loss)
    ####### start here ##########################
    all_prob = []
    all_num  = 0
    all_loss = np.zeros(2,np.float32)


    net.set_mode('test')
    for input, truth, index, cache in test_loader:
        print('\r',all_num, end='', flush=True)
        batch_size = len(index)
        all_num += batch_size

        input = input.cuda()
        with torch.no_grad():
            logit = net(input)
            prob  = F.sigmoid(logit)

            if 0: ##for debug
                truth = truth.cuda()
                loss  = net.criterion(logit, truth)
                dice  = net.metric(logit, truth)
                all_loss += batch_size*np.array(( loss.item(), dice.item(),))

        ##-----------------------------
        prob = prob.squeeze(1).data.cpu().numpy()
        prob = test_unaugment(prob)
        all_prob.append(prob)

        if 0: ##for debug

            os.makedirs(out_dir +'/test/%s/%s'%(split,augment), exist_ok=True)

            for b in range(batch_size):
                name    = test_dataset.ids[index[b]]
                predict = prob[b]
                image   = cache[b].image*255
                truth   = cache[b].mask
                image   = np.dstack([ image, image, image])


                overlay0 = draw_mask_overlay(predict, image, color=[0,0,255])
                overlay0 = draw_mask_to_contour_overlay(predict, overlay0, 2, color=[0,0,255])

                if truth==[]:
                    overlay1 = np.zeros((101,101,3),np.float32)
                else:
                    overlay1 = draw_mask_overlay(truth, image, color=[255,0,0])

                overlay = np.hstack([image, overlay0,overlay1])
                cv2.imwrite(out_dir +'/test/%s/%s/%s.png'%(split,augment,name),overlay*255)


                #image_show_norm('overlay',overlay,1,2)
                image_show('overlay',overlay,2)
                cv2.waitKey(0)

    print('\r',all_num, end='\n', flush=True)
    all_prob = np.concatenate(all_prob)
    all_prob = (all_prob*255).astype(np.uint8)
    np.save( out_dir +'/test/%s-%s.prob.uint8.npy'%(split,augment),all_prob)
    print(all_prob.shape)


    print('')
    assert(all_num == len(test_loader.sampler))
    all_loss  = all_loss/all_num
    print(all_loss)
    log.write('\n')

def run_submit(augment):

    if augment in ['null','flip']:

        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint8.npy'%(split,augment),
        ]
        csv_file = out_dir + '/test/%s-%s.csv'%(split,augment)


    if augment == 'aug2':
        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint8.npy'%(split,'null'),
            1, out_dir + '/test/%s-%s.prob.uint8.npy'%(split,'flip'),
        ]
        csv_file = out_dir + '/test/%s-%s.csv'%(split,augment)

    if augment == '10fold':
        augmentation = []
        for i in range():
            augmentation.append(1)
            augmentation.append(out_dir + "/fold%d/test/%s-%s.prob.uint8.npy"%(i, split,'null'))
            augmentation.append(1)
            augmentation.append(out_dir + "/fold%d/test/%s-%s.prob.uint8.npy"%(i, split,'flip'))
        
        csv_file = out_dir + "/average_10fold.csv"

    # csv_file = csv_file[:-4] + "_{}.csv".format(threshold)
    ##---------------------------------------

    #augments, csv_file = ['null','flip'], '/submit1_simple-valid0-300-aug.csv.gz'
    #augments, csv_file = ['flip'], '/submit1_simple-xxx-flip.csv.gz'
    #augments, csv_file = ['null'], '/submit1_simple-xxx-null.csv.gz'

    ##---------------------------------------

    #save
    augmentation = np.array(augmentation, dtype=object).reshape(-1,2)
    print(augmentation.shape)
    num_augments = len(augmentation)
    w, augment_file = augmentation[0]
    all_prob = w*np.load(augment_file).astype(np.float32)/255
    all_w = w
    for i in range(1, num_augments):
        w, augment_file = augmentation[i]
        prob = w*np.load(augment_file).astype(np.float32)/255
        all_prob += prob
        all_w += w
    all_prob /= all_w
    all_prob = all_prob>0.5
    print(all_prob.shape)


    #----------------------------

    split_file = '/data/jiuchao/TGS_Salt_Identification/data/split/' + split
    lines = read_list_from_file(split_file)


    id = []
    rle_mask = []
    for n, line in enumerate(lines):
        folder, name = line.split('/')
        id.append(name)

        if (all_prob[n].sum()<=0):
            encoding=''
        else:
            encoding = do_length_encode(all_prob[n])
        assert(encoding!=[])

        rle_mask.append(encoding)

    df = pd.DataFrame({ 'id' : id , 'rle_mask' : rle_mask}).astype(str)
    df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'])


############################################################################################
def run_local_leaderboard(augment):

    #-----------------------------------------------------------------------
    submit_file = out_dir + '/test/%s-%s.csv'%(split,augment)
    dump_dir = out_dir + '/test/%s-%s-dump'%(split,augment)
    os.makedirs(dump_dir, exist_ok=True)


    log = Logger()
    log.open(out_dir+'/test/log.submit.txt',mode='a')

    split_file = '/data/jiuchao/TGS_Salt_Identification/data/split/' + split
    lines = read_list_from_file(split_file)
    ids = [line.split('/')[-1] for line in lines]
    sorted(ids)


    df_submit = pd.read_csv(submit_file).set_index('id')
    df_submit = df_submit.fillna('')

    df_truth  = pd.read_csv('/data/jiuchao/TGS_Salt_Identification/data/train.csv').set_index('id')
    df_truth  = df_truth.loc[ids]
    df_truth  = df_truth.fillna('')

    N = len(df_truth)
    predict = np.zeros((N,101,101),np.bool)
    truth   = np.zeros((N,101,101),np.bool)

    for n in  range(N):
        id = ids[n]
        p  = df_submit.loc[id].rle_mask
        t  = df_truth.loc[id].rle_mask
        p  = do_length_decode(p, H=101, W=101, fill_value=1).astype(np.bool)
        t  = do_length_decode(t, H=101, W=101, fill_value=1).astype(np.bool)

        predict[n]=p
        truth[n]=t

        # if 0:
        #     image_p = predict[n].astype(np.uint8)*255
        #     image_t = truth[n]  .astype(np.uint8)*255
        #     image_show('image_p', image_p,2)
        #     image_show('image_t', image_t,2)
        #     cv2.waitKey(0)



    ##--------------
    precision, result, threshold = do_kaggle_metric(predict,truth, threshold=0.5)
    precision_mean = precision.mean()

    tp, fp, fn, tn_empty, fp_empty = result.transpose(1,2,0).sum(2)
    all = tp + fp + fn + tn_empty + fp_empty
    p   = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

    areas = np.sum(truth, axis=(1, 2)) / pow(101, 2)
    def area_to_class(a):
        for i in range(0, 11):
            if a*10 <= i:
                return i

    area_class = np.apply_along_axis(area_to_class, 1, areas[:, np.newaxis])

    log.write('\n')
    log.write('      |        |     \n')
    log.write('area  |  LB    |  num\n')
    log.write('---------------------\n')
    for i in range(0, 11):
        idx = np.where(area_class == i)[0]
        log.write('%0.2f |  %0.2f |  %5d\n' %(
            i/10.0, np.mean(precision[idx]), len(idx)
        ))


    log.write('\n')
    log.write('** %s ** \n'%augment)
    log.write('      |        |                                      |           empty          |         \n')
    log.write('th    |  prec  |      tp          fp          fn      |      tn          fp      |         \n')
    log.write('-------------------------------------------------------------------------------------------\n')
    for i, t in enumerate(threshold):
        log.write('%0.2f  |  %0.2f  |  %3d / %0.2f  %3d / %0.2f  %3d / %0.2f  |  %3d / %0.2f  %3d / %0.2f  | %5d\n'%(
            t, p[i],
            tp[i], tp[i]/all[i],
            fp[i], fp[i]/all[i],
            fn[i], fn[i]/all[i],
            tn_empty[i], tn_empty[i]/all[i],
            fp_empty[i], fp_empty[i]/all[i],
            all[i])
        )



    log.write('\n')
    log.write('num images :    %d\n'%N)
    log.write('LB score   : %0.5f\n'%(precision_mean))


    #--------------------------------------
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)
    p = predict>0.5
    t = truth>0.5
    intersection = t & p
    union        = t | p
    #iou = intersection.sum(1)/(union.sum(1)+EPS)
    log.write('iou        : %0.5f\n'%(intersection.sum()/(union.sum()+EPS)))

    return
    #exit(0)
    ## show --------------------------

    predicts = predict.reshape(-1,101,101).astype(np.float32)
    truths = truth.reshape(-1,101,101).astype(np.float32)
    for m,name in enumerate(ids):
        print('%s'%name)
        print('      |        |               |  empty  |      ')
        print('th    |  prec  |  tp  fp  fn   |  tn  fp |      ')
        print('------------------------------------------------')
        for i, t in enumerate(threshold):
            tp,fp,fn,fp_empty,tn_empty = result[m,:,i]
            p   = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)
            print('%0.2f  |  %0.2f  |   %d   %d   %d   |   %d   %d   '%(
                t, p, tp,fp,fn,fp_empty,tn_empty) )
        print(precision[m])
        print('')
        #----
        image_file = '/data/jiuchao/TGS_Salt_Identification/data/train/images/' + name +'.png'
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        #mask = mask>0

        predict = predicts[m]
        truth   = truths[m]

        #print(predict.sum())

        overlay0 = draw_mask_overlay(predict, image, color=[0,0,255])
        overlay0 = draw_mask_to_contour_overlay(predict, overlay0, 1, color=[0,0,255])
        overlay1 = draw_mask_overlay(truth,   image, color=[0,255,0])
        overlay1 = draw_mask_to_contour_overlay(truth,   overlay1, 1, color=[0,255,0])
        overlay2 = draw_mask_overlay(predict, None, color=[0,0,255])
        overlay2 = draw_mask_overlay(truth, overlay2, color=[0,255,0])

        draw_shadow_text(image,'%0.2f'%precision[m],(3,15),0.5,[255,255,255],1)

        overlay = np.hstack([image, overlay0, overlay1, overlay2])
        cv2.imwrite(dump_dir+'/%s.png'%name,overlay)
        image_show('overlay',overlay,2)
        cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # csv_files = []
    # csv_files.append('/data/jiuchao/TGS_Salt_Identification/Heng CherKeng/20180825/results/unet-5-scale-more-aug/fold0-b/test/list_test_18000-aug2.csv')
    # csv_files.append('/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet34_bn_256/test/list_test_18000-aug2.csv')
    # csv_files.append('/data/jiuchao/TGS_Salt_Identification/tsg_pytorch/results/resnet50_bn_256/test/list_test_18000-aug2.csv')

    # # ensemble(csv_files)
    for a in ['null', 'flip']:
        run_predict(a)
    # # # #exit(0)

    run_submit('aug2')


    # for a in ['null','flip', 'aug2']:
    #     run_submit(a)
    #     run_local_leaderboard(a)

    # print(lbs)
    # run_predict('aug2')
    # run_submit('aug2')
    # run_local_leaderboard()

    print('\nsucess!')
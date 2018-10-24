from common import *

DATA_DIR = '/data/jiuchao/TGS_Salt_Identification/data'
IMAGE_HEIGHT, IMAGE_WIDTH = 101, 101
HEIGHT, WIDTH = 128, 128

DY0, DY1, DX0, DX1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)

#--------------------------------
def null_augment(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())
    return image, mask, index, cache


def null_collate(batch):
    batch_size = len(batch)
    cache = []
    input = []
    truth = []
    index = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        index.append(batch[b][2])
        cache.append(batch[b][3])
    input = torch.from_numpy(np.array(input)).float().unsqueeze(1)

    if truth[0] != []:
        truth = torch.from_numpy(np.array(truth)).float().unsqueeze(1)

    return input, truth, index, cache


# def valid_augment(image, mask, index):
#     cache = Struct(image=image.copy(), mask=mask.copy())
#     image, mask = do_center_pad_to_factor2(image, mask, factor=32)
#     return image, mask, index, cache


#-------------------------------------------------
class TsgDataset(Dataset):
    def __init__(self, split, augment=null_augment, mode='train'):
        super(TsgDataset, self).__init__()

        self.split = split
        self.mode = mode
        self.augment = augment

        split_file = DATA_DIR + '/split/' + split
        lines = read_list_from_file(split_file)

        self.ids = []
        self.images = []
        for l in lines:
            folder, name = l.split('/')
            image_file = DATA_DIR + '/' + folder + '/images/' + name + '.png'
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            self.images.append(image)
            self.ids.append(name)

        self.masks = []
        if self.mode in ['train', 'valid']:
            for l in lines:
                folder, name = l.split('/')
                mask_file = DATA_DIR + '/' + folder + '/masks/' + name + '.png'
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
                self.masks.append(mask)
        elif self.mode in ['test']:
            self.masks = [[] for l in lines]
        
        #---------------------------------
        df = pd.read_csv(DATA_DIR + '/depths.csv')
        df = df.set_index('id')
        self.zs = df.loc[self.ids].z.values

        #-------
        print('\tTsgDataset')
        print('\tsplit            = %s'%split)
        print('\tlen(self.images) = %d'%len(self.images))
        print('')


    def __getitem__(self, index):
        image = self.images[index]
        mask  = self.masks[index]

        return self.augment(image, mask, index)

    def __len__(self):
        return len(self.images)






class DatasetCatalog(object):

    # dataset_root = "/Users/lizhixuan/PycharmProjects/amodal_dataset/"
    dataset_root = "/datasets/lzx/amodal/"
    dataset_attrs = {
        'CocoTrain': {
            'id': 'coco',
            'data_root': 'data/coco/train2017',
            'ann_file': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'CocoVal': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'test'
        },
        'CocoMini': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'mini'
        },
        'CocoTest': {
            'id': 'coco_test',
            'data_root': 'data/coco/test2017',
            'ann_file': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'CityscapesTrain': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'CityscapesVal': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'CityscapesCocoVal': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'CityCocoBox': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_box_val.json',
            'split': 'val'
        },
        'CityscapesMini': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'mini'
        },
        'CityscapesTest': {
            'id': 'cityscapes_test',
            'data_root': 'data/cityscapes/leftImg8bit/test'
        },
        'SbdTrain': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'SbdVal': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'SbdMini': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'mini'
        },
        'VocVal': {
            'id': 'voc',
            'data_root': 'data/voc/JPEGImages',
            'ann_file': 'data/voc/annotations/voc_val_instance.json',
            'split': 'val'
        },
        'KinsTrain': {
            'id': 'kins',
            'data_root': dataset_root + 'KINS/images/train',
            'ann_file': dataset_root + 'KINS/annotations/KINS_train.json',
            'split': 'train'
        },
        'KinsVal': {
            'id': 'kins',
            'data_root': dataset_root + 'KINS/images/test',
            'ann_file': dataset_root + 'KINS/annotations/KINS_val.json',
            'split': 'val'
        },
        'KinsMini': {
            'id': 'kins',
            'data_root': dataset_root + 'KINS/images/test',
            'ann_file': dataset_root + 'KINS/annotations/KINS_val.json',
            'split': 'mini'
        },
        'cocoa_train': {
            'id': 'cocoa',
            'data_root': dataset_root + 'COCOA/images/train',
            'ann_file': dataset_root + 'COCOA/annotations/COCOA_train.json',
            'split': 'train'
        },
        'cocoa_test': {
            'id': 'cocoa',
            'data_root': dataset_root + 'COCOA/images/val',
            'ann_file': dataset_root + 'COCOA/annotations/COCOA_val.json',
            'split': 'val'
        },
        'BSDSA_train': {
            'id': 'BSDSA',
            'data_root': dataset_root + 'BSDSA/images/train',
            'ann_file': dataset_root + 'BSDSA/annotations/BSDSA_train.json',
            'split': 'train'
        },
        'BSDSA_val': {
            'id': 'BSDSA',
            'data_root': dataset_root + 'BSDSA/images/val',
            'ann_file': dataset_root + 'BSDSA/annotations/BSDSA_val.json',
            'split': 'val'
        },
        'BSDSA_test': {
            'id': 'BSDSA',
            'data_root': dataset_root + 'BSDSA/images/test',
            'ann_file': dataset_root + 'BSDSA/annotations/BSDSA_test.json',
            'split': 'test'
        },
        'BSDSA_trainval': {
            'id': 'BSDSA',
            'data_root': dataset_root + 'BSDSA/images/trainval',
            'ann_file': dataset_root + 'BSDSA/annotations/BSDSA_trainval.json',
            'split': 'trainval'
        },
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()


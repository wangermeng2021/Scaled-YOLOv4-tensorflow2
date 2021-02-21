
from generator.voc_generator import VocGenerator
from generator.coco_generator import CocoGenerator
def get_generator(args):
    if args.dataset_type == 'voc':
        train_dataset = VocGenerator(args, mode=0)
        valid_dataset = VocGenerator(args, mode=1)
        pred_dataset  = VocGenerator(args, mode=2)
    elif args.dataset_type == 'coco':
        train_dataset = CocoGenerator(args, mode=0)
        valid_dataset = CocoGenerator(args, mode=1)
        pred_dataset  = CocoGenerator(args, mode=2)
    else:
        raise ValueError("{} is invalid!".format(args.dataset_type))
    # return train_dataset, valid_dataset, pred_dataset
    return train_dataset, valid_dataset, pred_dataset
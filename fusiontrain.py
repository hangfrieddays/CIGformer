from models.trainer import PromoteFusionTrainer
import logging
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_parse_args():
    parser = argparse.ArgumentParser(description='Training of CIGformer') 
    parser.add_argument('--pretrained', default=False, type=bool, help='use pretrained') 
    parser.add_argument('--log_pth', default='./records/CIGformer/train_detail.txt', type=str, help='log path')
    parser.add_argument('--log_name', default='CIGformer', type=str, help='promote fusion trainer log')
    parser.add_argument('--config_pth', default='records/CIGformer/config.yml', type=str, help='config path of training details')

    return parser

if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()

    trainer = PromoteFusionTrainer(
        config_file=args.config_pth,
        log_name=args.log_name,
        log_level=logging.INFO,
        log_pth=args.log_pth,
        use_pretrained=args.pretrained
    )
    trainer.train()

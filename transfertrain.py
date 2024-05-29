import argparse
import logging
from models.trainer import TransferTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration for TransferTrainer')
    parser.add_argument('--config_file', default="records/transfer/config.yml", type=str, help='Path to the configuration file')
    parser.add_argument('--log_name', default='transfer trainer log', type=str, help='Name for the log file')
    parser.add_argument('--log_pth', default='records/transfer/transfer.txt', type=str, help='Path to the log file')
    parser.add_argument('--use_pretrained', default=False, type=bool, help='Flag to determine if pretrained model should be used')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    trainer = TransferTrainer(
        config_file=args.config_file,
        log_name=args.log_name,
        log_level=logging.INFO,
        log_pth=args.log_pth,
        use_pretrained=args.use_pretrained
    )
    trainer.train()

"""
    定义了如何设置logger并且输出到特定位置
    可用于在模型的训练全过程中记录模型的状态

"""
import logging
import time
import os

class Logger(object):
    def __init__(self, log_pth, log_level, log_name):
        # firstly, create a logger
        self.__logger = logging.getLogger(log_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler = logging.FileHandler(log_pth, mode = 'w', encoding='utf-8')
        console_handler = logging.StreamHandler()
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

# if __name__ == '__main__':
#     # log = Logger(log_pth='./log.txt', log_level=logging.INFO, logger_name='log!!')
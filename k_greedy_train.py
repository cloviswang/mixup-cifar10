import argparse
import csv
import logging
import os
import time
import torch

global best_acc_a
global best_acc_b


class KGreedyTrain(object):

    def __init__(self):
        self.init_log()
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--turn-epoch', default=5, type=int, help='turn epoch')
        args = parser.parse_args()
        self.turn_epoch = args.turn_epoch
        self.gpu_id = args.gpu
        self.aug_method_1 = 'mixup'
        self.aug_method_2 = 'ricap'
        self.train_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def init_log(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        log_path = 'Logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)

    def save_csv(self, epoch, best_acc_a, best_acc_b, method):
        logname = f'results/log_k_greedy_{self.turn_epoch}_{self.train_name}'
        logname += '.csv'
        if not os.path.exists(logname):
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(
                    ['start epoch', 'end epoch', f'{self.aug_method_1}', f'{self.aug_method_2}', 'method'])
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch * self.turn_epoch, (epoch + 1) * self.turn_epoch, best_acc_a, best_acc_b, method])

    def main(self):
        max_epoch = 200
        train_cmd = f"CUDA_VISIBLE_DEVICES={self.gpu_id}, python train.py --name {self.train_name} " \
                    f"--epoch {self.turn_epoch} --aug-method-1 {self.aug_method_1} --aug-method-2 {self.aug_method_2}"
        self.logger.info(train_cmd)
        result = os.system(train_cmd)
        if result != 0:
            self.logger.error("error!")
            return "error"

        for epoch in range(1, int(max_epoch / self.turn_epoch)):
            # 拆分
            self.logger.info(f"____________复制两份checkpoint_____________")
            cp_cmd = f"cp checkpoint/ckpt.t7{self.train_name}_0 checkpoint/ckpt.t7{self.train_name}_a_0"
            train_name_a = f"{self.train_name}_a"
            result = os.system(cp_cmd)
            if result != 0:
                self.logger.error("error!")
                return "error"
            mv_cmd = f"mv checkpoint/ckpt.t7{self.train_name}_0 checkpoint/ckpt.t7{self.train_name}_b_0"
            train_name_b = f"{self.train_name}_b"
            result = os.system(mv_cmd)
            if result != 0:
                self.logger.error("error!")
                return "error"

            # 分别训练
            self.logger.info(f"训练方法: {self.aug_method_1}")
            train_a_cmd = f"CUDA_VISIBLE_DEVICES={self.gpu_id}, python train.py --name {train_name_a} " \
                          f"--epoch {(epoch + 1) * self.turn_epoch} " \
                          f"--aug-method-1 {self.aug_method_1} --aug-method-2 {self.aug_method_1} --resume -r"
            self.logger.info(train_a_cmd)
            result = os.system(train_a_cmd)
            if result != 0:
                self.logger.error("error!")
                return "error"
            self.logger.info(f"训练方法: {self.aug_method_2}")
            train_b_cmd = f"CUDA_VISIBLE_DEVICES={self.gpu_id}, python train.py --name {train_name_b} " \
                          f"--epoch {(epoch + 1) * self.turn_epoch} " \
                          f"--aug-method-1 {self.aug_method_2} --aug-method-2 {self.aug_method_2} --resume -r"
            self.logger.info(train_b_cmd)
            result = os.system(train_b_cmd)
            if result != 0:
                self.logger.error("error!")
                return "error"

            # 合并
            self.logger.info(f"________比较best_acc, 选择最好的一个________")
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint_a = torch.load('./checkpoint/ckpt.t7' + train_name_a + "_0")
            best_acc_a = checkpoint_a['acc']
            self.logger.info(f"{self.aug_method_1} best_acc={best_acc_a}")
            checkpoint_b = torch.load('./checkpoint/ckpt.t7' + train_name_b + "_0")
            best_acc_b = checkpoint_b['acc']
            self.logger.info(f"{self.aug_method_2} best_acc={best_acc_b}")
            if best_acc_a > best_acc_b or (best_acc_a == best_acc_b and epoch % 2 == 0):
                self.logger.info(f"第{epoch * self.turn_epoch}到{(1 + epoch) * self.turn_epoch}个迭代，"
                                 f"选择{self.aug_method_1}。(best_acc:{best_acc_a}>{best_acc_b})")
                self.save_csv(epoch, best_acc_a, best_acc_b, self.aug_method_1)
                mv_cmd = f"mv checkpoint/ckpt.t7{self.train_name}_a_0 checkpoint/ckpt.t7{self.train_name}_0"
                result = os.system(mv_cmd)
                if result != 0:
                    self.logger.error("error!")
                    return "error"
            else:
                self.logger.info(f"第{epoch * self.turn_epoch}到{(1 + epoch) * self.turn_epoch}个迭代，"
                                 f"选择{self.aug_method_2}。(best_acc:{best_acc_a}<{best_acc_b})")
                mv_cmd = f"mv checkpoint/ckpt.t7{self.train_name}_b_0 checkpoint/ckpt.t7{self.train_name}_0"
                self.save_csv(epoch, best_acc_a, best_acc_b, self.aug_method_2)
                result = os.system(mv_cmd)
                if result != 0:
                    self.logger.error("error!")
                    return "error"


if __name__ == "__main__":
    train = KGreedyTrain()
    train.main()

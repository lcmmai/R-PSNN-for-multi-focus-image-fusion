import argparse
import os
import Train
import Test

def main(config):
    if config.mode == 'train':
        Train.train(config)
    elif config.mode == 'test':
        Test.test(config)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img1_path', type=str, default='./Dataset/img1/', help="the first multi-focus image.")
    parser.add_argument('--img2_path', type=str, default='./Dataset/img2/', help="the second multi-focus image.")
    parser.add_argument('--groundtruth_path', type=str, default='./Dataset/gt/', help="the groundtruth.")

    parser.add_argument('--img1_test_path', type=str, default='./Testset/img1/', help="the first multi-focus image.")
    parser.add_argument('--img2_test_path', type=str, default='./Testset/img2/', help="the second multi-focus image.")

    parser.add_argument('--img_save_path', type=str, default='./Test_result/', help="the second multi-focus image.")

    parser.add_argument('--model_save_path', type=str, default='./model/', help="the saved models.")
    parser.add_argument('--mode', type=str, default='test', help="train or test.")

    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate.")
    parser.add_argument('--beta1', type=float, default=0.9, help="the first parameter in Adam.")
    parser.add_argument('--beta2', type=float, default=0.999, help="the second parameter in Adam.")

    parser.add_argument('--sum_epochs', type=int, default=60)
    parser.add_argument('--decay_epochs', type=int, default=5, help="halve the lr every decay_epochs.")
    parser.add_argument('--save_epochs', type=int, default=1, help="save model every save_epochs.")
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--cuda_id', type=str, default='0')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    main(config)
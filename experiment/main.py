import warnings
import argparse
warnings.filterwarnings('ignore')

from . import utils, data, features, train

def main():
    """主函数，根据命令行参数调度实验流程"""
    parser = argparse.ArgumentParser(description="实验流程控制器")
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的命令')

    # --- 特征生成命令 ---
    parser_gen = subparsers.add_parser('gen-feats', help='生成或更新特征')
    parser_gen.add_argument(
        '--funcs', 
        nargs='*', 
        default=None,
        help='要生成/更新的特征函数名称列表。如果未提供，则生成所有已注册的特征。'
    )

    # --- 特征删除命令 ---
    parser_del = subparsers.add_parser('del-feats', help='删除特征')
    parser_del.add_argument(
        '--funcs', 
        nargs='+', 
        required=True,
        help='要删除的特征函数名称列表。'
    )

    # --- 训练命令 ---
    parser_train = subparsers.add_parser('train', help='使用已有的特征文件进行训练')

    args = parser.parse_args()

    # 设置日志
    logger = utils.setup_logger()
    logger.info("=======================================")
    logger.info(f"========== Running Command: {args.command} ==========")
    logger.info("=======================================")

    if args.command == 'gen-feats':
        logger.info("开始特征生成流程...")
        X_train, _ = data.load_data()
        features.generate_features(X_train, funcs_to_run=args.funcs)
        logger.info("特征生成完成。")

    elif args.command == 'del-feats':
        logger.info("开始特征删除流程...")
        features.delete_features(funcs_to_delete=args.funcs)
        logger.info("特征删除完成。")

    elif args.command == 'train':
        logger.info("开始模型训练流程...")
        models, oof_auc = train.train_and_evaluate()
        if oof_auc is not None:
            logger.info(f"模型训练完成。Final OOF AUC: {oof_auc:.5f}")
        else:
            logger.info("模型训练因故中止。")

if __name__ == '__main__':
    main() 
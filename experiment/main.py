import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
import argparse
import sys
import logging
from . import utils, data, features, train, config
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", FutureWarning)

def main():
    """主函数，根据命令行参数调度实验流程"""
    parser = argparse.ArgumentParser(description="实验流程控制器")
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的命令')

    # --- 特征生成命令 ---
    parser_gen = subparsers.add_parser('gen-feats', help='生成或更新特征，并创建一个新的带时间戳的特征文件')
    parser_gen.add_argument('--funcs', nargs='*', default=None, help='要生成/更新的特征函数名列表。如果为空，则运行所有注册的函数。')
    parser_gen.add_argument('--trans', nargs='*', default=None, help='要生成/更新的时序变换函数名列表。如果为空，则运行所有注册的函数。')
    parser_gen.add_argument('--base-file', type=str, default=None, help='可选，指定一个基础特征文件名进行更新。如果为空，则使用最新的特征文件。')

    # --- 特征删除命令 ---
    parser_del = subparsers.add_parser('del-feats', help='从指定的特征文件中删除特征，并创建一个新的带时间戳的文件')
    parser_del.add_argument('--base-file', type=str, required=True, help='必须指定一个基础特征文件名进行操作。')
    group = parser_del.add_mutually_exclusive_group(required=True)
    group.add_argument('--funcs', nargs='+', help='要删除的特征函数名称列表。')
    group.add_argument('--cols', nargs='+', help='要删除的特定特征列名列表。')

    # --- 特征交互项生成命令 ---
    parser_inter = subparsers.add_parser('gen-interactions', help='根据特征重要性文件生成交互特征')
    parser_inter.add_argument('--importance-file', type=str, required=False, help='特征重要性文件路径 (e.g., permutation_importance.tsv).')
    parser_inter.add_argument('--base-file', type=str, default=None, help='可选，指定一个基础特征文件名进行更新。如果为空，则使用最新的特征文件。')
    parser_inter.add_argument('--add', action='store_true', help='创建加法交互项。')
    parser_inter.add_argument('--sub', action='store_true', help='创建减法交互项。')
    parser_inter.add_argument('--div', action='store_true', help='创建除法交互项。')
    parser_inter.add_argument('--no-mul', dest='mul', action='store_false', help='不创建乘法交互项(默认为创建)。')

    # --- 特征筛选命令 ---
    parser_filter = subparsers.add_parser('filter', help='特征筛选工具')
    filter_subparsers = parser_filter.add_subparsers(dest='filter_method', required=True, help='筛选方法')
    # 相关性筛选子命令
    parser_corr = filter_subparsers.add_parser('corr', help='根据特征相关性筛选特征并导出')
    parser_corr.add_argument('--feature-file', type=str, default=None, help='可选，指定特征文件名。如果为空，则使用最新的特征文件。')
    # 置换重要性筛选子命令
    parser_perm_imp = filter_subparsers.add_parser('perm-imp', help='根据特征重要性阈值筛选特征并导出')
    parser_perm_imp.add_argument('--train-version', type=str, help='训练输出文件夹名，例如 train_20250719_174900_auc_0_76876')
    parser_perm_imp.add_argument('--feature-file', type=str, help='特征文件名，用于创建输出目录')
     
    # --- 数据增强命令 ---
    parser_enhance = subparsers.add_parser('data-aug', help='应用数据增强并保存增强数据')
    parser_enhance.add_argument('--augs', nargs='*', default=None, help='要应用的数据增强函数名列表。如果为空，则运行所有注册的增强函数。')

    # --- 训练命令 ---
    parser_train = subparsers.add_parser('train', help='使用特征文件进行训练')
    parser_train.add_argument('--feature-file', type=str, default=None, help='可选，指定用于训练的特征文件名。如果为空，则使用最新的特征文件。')
    parser_train.add_argument('--train-data-ids', nargs='*', default=["0"], help='可选，指定用于训练的数据ID列表。如果为空，则使用原始数据。')
    parser_train.add_argument('--perm-imp', action='store_true', help='是否计算permutation importance。')
    parser_train.add_argument('--save-oof', action='store_true', help='是否保存OOF预测文件。')
    parser_train.add_argument('--save-model', action='store_true', help='是否保存训练好的模型文件。')

    # --- 超参调优命令 ---
    parser_tune = subparsers.add_parser('tune', help='使用Optuna进行超参调优')
    parser_tune.add_argument('--feature-file', type=str, default=None, help='可选，指定用于训练的特征文件名。如果为空，则使用最新的特征文件。')
    parser_tune.add_argument('--train-data-ids', nargs='*', default=["0"], help='可选，指定用于训练的数据ID列表。如果为空，则使用原始数据。')
    parser_tune.add_argument('--n-trials', type=int, default=50, help='Optuna 试验次数，默认50')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    # 根据命令选择 logger
    log_file_path = None # 初始化
    if args.command in ['gen-feats', 'del-feats', 'gen-interactions', 'data-aug']:
        logger, log_file_path = utils.get_logger('FeatureEng', config.FEATURE_LOG_DIR)
    else: # train
        logger, log_file_path = utils.get_logger('Training', config.TRAINING_LOG_DIR)

    logger.info(f"========== Running Command: {args.command} ==========")
    logger.info(f"Args: {vars(args)}")
    logger.info("=======================================")

    if args.command == 'data-aug':
        from . import data
        data.logger = logger
        data.apply_data_enhancement(func_names=args.augs)

    elif args.command == 'gen-feats':
        from . import data, features
        features.logger = logger
        data.logger = logger
        X_data, y_data = data.load_data(enhancement_ids=config.ENHANCEMENT_IDS)
        features.generate_features(X_data, funcs_to_run=args.funcs, trans_to_run=args.trans, base_feature_file=args.base_file)

    elif args.command == 'del-feats':
        from . import features
        features.logger = logger
        features.delete_features(
            base_feature_file=args.base_file,
            funcs_to_delete=args.funcs, 
            cols_to_delete=args.cols
        )

    elif args.command == 'gen-interactions':
        from . import interactions
        interactions.logger = logger
        # 确保 features 模块的 logger 也被设置
        from . import features
        features.logger = logger
        interactions.generate_interaction_features(
            importance_file_path=args.importance_file,
            base_feature_file=args.base_file,
            create_mul=args.mul,
            create_add=args.add,
            create_sub=args.sub,
            create_div=args.div
        )

    elif args.command == 'filter':
        if args.filter_method == 'corr':
            from . import filter
            filter.corr_filter(
                feature_file=args.feature_file
            )
        elif args.filter_method == 'perm-imp':
            from . import filter
            filter.perm_imp_filter(
                train_version=args.train_version, 
                feature_file=args.feature_file
            )

    elif args.command == 'train':
        from . import train, features, data
        data.logger = logger
        train.logger = logger
        features.logger = logger
        models, oof_auc = train.train_and_evaluate(
            feature_file_name=args.feature_file,
            data_ids=args.train_data_ids,
            save_oof=args.save_oof,
            save_model=args.save_model,
            perm_imp=args.perm_imp
        )
        
        # 训练成功后重命名日志文件
        if oof_auc is not None:
            # 1. 关闭 logger 的文件处理器，释放对文件的占用
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            # 2. 重命名文件
            auc_str = f"{oof_auc:.5f}".replace('.', '_')
            new_log_path = log_file_path.with_name(f"{log_file_path.stem}_auc_{auc_str}{log_file_path.suffix}")
            try:
                log_file_path.rename(new_log_path)
                print(f"Log file renamed to: {new_log_path.name}")
            except OSError as e:
                print(f"Error renaming log file: {e}")

    elif args.command == 'tune':
        from . import train, features, data
        data.logger = logger
        train.logger = logger
        features.logger = logger
        best_oof_auc = train.tune_hyperparameter(
            feature_file_name=args.feature_file,
            data_ids=args.train_data_ids,
            n_trials=args.n_trials,
        )
        
        # 训练成功后重命名日志文件
        if best_oof_auc is not None:
            # 1. 关闭 logger 的文件处理器，释放对文件的占用
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            # 2. 重命名文件
            auc_str = f"{best_oof_auc:.5f}".replace('.', '_')
            new_log_path = log_file_path.with_name(f"{log_file_path.stem}_auc_{auc_str}{log_file_path.suffix}")
            try:
                log_file_path.rename(new_log_path)
                print(f"Log file renamed to: {new_log_path.name}")
            except OSError as e:
                print(f"Error renaming log file: {e}")

if __name__ == '__main__':
    main()
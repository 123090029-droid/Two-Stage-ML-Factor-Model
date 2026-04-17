import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List

import tensorflow as tf
from tensorflow.keras import layers, Model

def xs_zscore(x):
    '''截面标准化'''
    result = x.sub(x.mean(axis=1), axis=0)
    stds = x.std(axis=1)
    mask = stds > 0
    result[mask] = result[mask].div(stds[mask], axis=0)
    return result

def OLS_regression(ret: pd.DataFrame, f_list: List[pd.DataFrame], train_period):
    # 对齐数据时间范围
    common_idx = ret.index.intersection(f_list[0].index)
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    train_start, train_end = train_period[0], train_period[1]
    
    train_mask = (common_idx >= train_start) & (common_idx <= train_end)
    train_dates = common_idx[train_mask]
    
    X_list, y_list = [], []
    
    for date in train_dates[:-1]:
        next_date = train_dates[train_dates.get_loc(date) + 1]
        
        # 获取因子数据
        X_date = pd.concat([f_df.loc[date] for f_df in f_list], axis=1)
        X_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 获取下一期收益
        y_date = ret.loc[next_date]
        
        # 合并并清理数据
        data = pd.concat([X_date, y_date], axis=1)
        data.columns = list(X_date.columns) + ['ret']
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) > 0:
            X_list.append(data.iloc[:, :-1])
            y_list.append(data.iloc[:, -1])
    
    # 合并所有截面数据
    X_train = pd.concat(X_list)
    y_train = pd.concat(y_list)
    
    # 添加常数项
    X_train = sm.add_constant(X_train)
    
    # 训练模型
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())
    
    return model

def linear_predict_returns(model, f_list, backtest_period):
    # 对齐因子数据时间范围
    common_idx = f_list[0].index
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    backtest_start, backtest_end = backtest_period[0], backtest_period[1]
    
    backtest_mask = (common_idx >= backtest_start) & (common_idx <= backtest_end)
    backtest_dates = common_idx[backtest_mask]
    
    predictions = {}
    
    # 获取模型参数数量
    n_params = len(model.params)
    n_factors = n_params - 1  # 减去常数项
    
    for date in backtest_dates:
        # 获取所有因子在该日期的共同股票
        common_stocks = set(f_list[0].loc[date].index)
        for f_df in f_list[1:]:
            common_stocks = common_stocks.intersection(f_df.loc[date].index)
        common_stocks = sorted(list(common_stocks))
        
        if len(common_stocks) == 0:
            continue
            
        # 构建特征矩阵，只使用共同股票
        X_date = np.column_stack([f_df.loc[date, common_stocks].values for f_df in f_list])
        
        # 处理缺失值和异常值
        valid_mask = ~(np.any(np.isinf(X_date), axis=1) | np.any(np.isnan(X_date), axis=1))
        X_clean = X_date[valid_mask]
        
        if len(X_clean) > 0:
            # 手动创建带常数项的特征矩阵
            X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
            
            # 确保维度匹配
            if X_with_const.shape[1] == n_params:
                # 直接使用模型参数进行预测
                pred = np.dot(X_with_const, model.params)
                
                # 创建Series保持股票代码索引
                valid_stocks = [common_stocks[i] for i in range(len(common_stocks)) if valid_mask[i]]
                pred_series = pd.Series(pred, index=valid_stocks)
                predictions[date] = pred_series
            else:
                print(f"警告: 在日期 {date} 特征维度不匹配: 期望 {n_params}, 实际 {X_with_const.shape[1]}")
    
    predicted_returns = pd.DataFrame(predictions).T
    return predicted_returns

def calculate_ic_analysis(ret_hat, ret, pre_sample, in_sample, post_sample):
    # 对齐数据，只保留共同日期和股票
    common_dates = ret_hat.index.intersection(ret.index)
    ret_hat_aligned = ret_hat.loc[common_dates]
    ret_aligned = ret.loc[common_dates]
    
    # 定义样本期间
    pre_start, pre_end = pre_sample
    in_start, in_end = in_sample
    post_start, post_end = post_sample
    
    # 计算各期间的IC序列
    ic_results = {}
    
    for period_name, period_range in [
        ('Pre-sample', (pre_start, pre_end)),
        ('In-sample', (in_start, in_end)),
        ('Post-sample', (post_start, post_end))
    ]:
        start, end = period_range
        period_dates = common_dates[(common_dates >= start) & (common_dates <= end)]
        
        ic_series = []
        ic_dates = []
        
        for date in period_dates:
            # 获取当天的预测和实际收益率
            pred_returns = ret_hat_aligned.loc[date]
            actual_returns = ret_aligned.loc[date]
            
            # 对齐股票，只保留共同股票
            common_stocks = pred_returns.index.intersection(actual_returns.index)
            pred_common = pred_returns[common_stocks]
            actual_common = actual_returns[common_stocks]
            
            # 移除缺失值
            valid_mask = ~(np.isnan(pred_common) | np.isnan(actual_common))
            pred_valid = pred_common[valid_mask]
            actual_valid = actual_common[valid_mask]
            
            if len(pred_valid) > 1:  # 至少需要2个点计算相关系数
                ic, _ = pearsonr(pred_valid, actual_valid)
                ic_series.append(ic)
                ic_dates.append(date)
        
        ic_results[period_name] = {
            'ic_series': pd.Series(ic_series, index=ic_dates),
            'dates': ic_dates
        }
    
    # 计算统计量
    stats = {}
    for period_name, period_data in ic_results.items():
        ic_series = period_data['ic_series']
        if len(ic_series) > 0:
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std != 0 else 0
            
            stats[period_name] = {
                'IC Mean': ic_mean,
                'IC Std': ic_std,
                'ICIR': icir,
                'IC Series': ic_series
            }
    
    # 绘制IC累积曲线
    plt.figure(figsize=(9, 6))
    colors = ['blue', 'green', 'red']
    
    # 计算连续累积IC
    cumulative_base = 0
    for i, (period_name, period_stats) in enumerate(stats.items()):
        ic_series = period_stats['IC Series']
        cumulative_ic = ic_series.cumsum() + cumulative_base
        
        plt.plot(ic_series.index, cumulative_ic, 
                label=f'{period_name} (IC Mean: {period_stats["IC Mean"]:.4f}, ICIR: {period_stats["ICIR"]:.4f})',
                color=colors[i], linewidth=2)
        
        # 更新累积基准值为当前期的最后一个累积IC值
        if len(cumulative_ic) > 0:
            cumulative_base = cumulative_ic.iloc[-1]
    
    plt.title('Continuous Cumulative IC Curve by Sample Period', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative IC', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 打印统计结果
    """
    print("IC Analysis Results:")
    print("=" * 60)
    for period_name, period_stats in stats.items():
        print(f"{period_name}:")
        print(f"  IC Mean: {period_stats['IC Mean']:.6f}")
        print(f"  IC Std:  {period_stats['IC Std']:.6f}")
        print(f"  ICIR:    {period_stats['ICIR']:.6f}")
        print(f"  Observations: {len(period_stats['IC Series'])}")
        print("-" * 40)
    """
    
    return stats

def correlation_loss(y_true, y_pred):
    """自定义相关系数损失函数 - 最大化IC"""
    # 中心化
    y_true_centered = y_true - tf.reduce_mean(y_true, axis=0)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=0)
    
    # 计算协方差
    covariance = tf.reduce_mean(y_true_centered * y_pred_centered, axis=0)
    
    # 计算标准差
    y_true_std = tf.math.reduce_std(y_true, axis=0)
    y_pred_std = tf.math.reduce_std(y_pred, axis=0)
    
    # 计算相关系数
    correlation = covariance / (y_true_std * y_pred_std + 1e-8)

    return 1 - correlation

def create_keras_model(input_dim, output_dim, hidden_layer_sizes, activation, learning_rate=0.001, loss_type='MSE'):
    """创建Keras模型"""
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for units in hidden_layer_sizes:
        model.add(layers.Dense(units, activation=activation))
    
    model.add(layers.Dense(output_dim, activation='linear'))
    
    # 选择损失函数
    if loss_type.upper() == 'IC':
        loss_func = correlation_loss
    else:
        loss_func = 'mse'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_func,
        metrics=['mse']
    )
    
    return model

def train_outter_mlp_model(ret, f_list, train_period, epochs,
                   hidden_layer_sizes=(100,), activation='relu', 
                   solver='adam', alpha=0.0001, random_state=42,
                   loss_function='MSE'):
    """
    使用MLP模型训练因子收益率预测模型
    """
    tf.keras.utils.set_random_seed(random_state)
    np.random.seed(random_state)
    # 对齐数据时间范围
    common_idx = ret.index.intersection(f_list[0].index)
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    train_start, train_end = train_period[0], train_period[1]
    
    train_mask = (common_idx >= train_start) & (common_idx <= train_end)
    train_dates = common_idx[train_mask]
    
    X_list, y_list = [], []
    
    for date in train_dates[:-1]:
        next_date = train_dates[train_dates.get_loc(date) + 1]
        
        # 获取因子数据
        X_date = pd.concat([f_df.loc[date] for f_df in f_list], axis=1)
        X_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 获取下一期收益
        y_date = ret.loc[next_date]
        
        # 合并并清理数据
        data = pd.concat([X_date, y_date], axis=1)
        data.columns = list(X_date.columns) + ['ret']
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) > 0:
            X_list.append(data.iloc[:, :-1])
            y_list.append(data.iloc[:, -1])
    
    # 合并所有截面数据
    X_train = pd.concat(X_list)
    y_train = pd.concat(y_list)
    
    X_train_values = X_train.values
    y_train_values = y_train.values.reshape(-1, 1)
    
    # 创建Keras模型
    model = create_keras_model(
        input_dim=X_train_values.shape[1],
        output_dim=1,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        loss_type=loss_function
    )
    
    # 训练模型
    print(f"Training MLP model with {loss_function} loss...")
    history = model.fit(
        X_train_values, y_train_values,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # 计算训练得分
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_train_values, verbose=0)
    train_score = r2_score(y_train_values, y_pred)
    print(f"MLP Model Training R² Score: {train_score:.4f}")
    
    # 返回包含模型和标准化器的字典
    return {
        'model': model,
        'type': 'keras_outer'
    }

def mlp_predict_returns(model, f_list, backtest_period):
    backtest_start, backtest_end = backtest_period
    
    # 对齐因子数据时间范围
    common_idx = f_list[0].index
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    backtest_mask = (common_idx >= backtest_start) & (common_idx <= backtest_end)
    backtest_dates = common_idx[backtest_mask]
    
    predictions = {}
    
    print("Predicting returns...")
    for date in backtest_dates:
        # 获取当天因子数据
        X_date = pd.concat([f_df.loc[date] for f_df in f_list], axis=1)
        X_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 清理数据
        X_clean = X_date.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(X_clean) > 0:
            X_values = X_clean.values
            # 使用Keras模型预测
            pred = model['model'].predict(X_values, verbose=0)
            
            # 处理输出形状
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.flatten()
            
            # 创建Series保持股票代码索引
            pred_series = pd.Series(pred, index=X_clean.index)
            predictions[date] = pred_series
    
    predicted_returns = pd.DataFrame(predictions).T
    return predicted_returns

def train_inner_mlp_model(f_list, train_period, epochs,
                   hidden_layer_sizes=(100,), activation='relu', 
                   solver='adam', alpha=0.0001, random_state=42,
                   loss_function='MSE'):
    tf.keras.utils.set_random_seed(random_state)
    np.random.seed(random_state)
    # 对齐数据时间范围
    common_idx = f_list[0].index
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    train_start, train_end = train_period[0], train_period[1]
    train_mask = (common_idx >= train_start) & (common_idx <= train_end)
    train_dates = common_idx[train_mask]
    
    X_list, y_list = [], []
    
    for i in range(len(train_dates) - 1):
        current_date = train_dates[i]
        next_date = train_dates[i + 1]
        
        # 获取当期因子数据作为输入
        X_date = pd.concat([f_df.loc[current_date] for f_df in f_list], axis=1)
        X_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 获取下一期因子数据作为输出
        y_date = pd.concat([f_df.loc[next_date] for f_df in f_list], axis=1)
        y_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 合并并清理数据
        data = pd.concat([X_date, y_date], axis=1)
        data.columns = list(X_date.columns) + [f'target_{i}' for i in range(len(f_list))]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) > 0:
            X_list.append(data.iloc[:, :len(f_list)])
            y_list.append(data.iloc[:, len(f_list):])
    
    # 合并所有截面数据
    if len(X_list) == 0:
        print("警告: 没有有效的训练数据")
        return None
        
    X_train = pd.concat(X_list)
    y_train = pd.concat(y_list)
    
    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    
    X_train_values = X_train.values
    y_train_values = y_train.values
    
    # 创建Keras多输出模型
    model = create_keras_model(
        input_dim=X_train_values.shape[1],
        output_dim=y_train_values.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        loss_type=loss_function
    )
    
    # 训练模型
    print(f"Training inner MLP model with {loss_function} loss...")
    history = model.fit(
        X_train_values, y_train_values,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # 计算训练得分
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_train_values, verbose=0)
    train_score = r2_score(y_train_values, y_pred)
    print(f"多输出MLP模型训练R² Score: {train_score:.4f}")
    
    # 返回包含模型和标准化器的字典
    return {
        'model': model,
        'type': 'keras_inner'
    }

def inner_mlp_predict_returns(model, f_list, backtest_period):
    backtest_start, backtest_end = backtest_period
    
    # 对齐因子数据时间范围
    common_idx = f_list[0].index
    for f_df in f_list[1:]:
        common_idx = common_idx.intersection(f_df.index)
    
    backtest_mask = (common_idx >= backtest_start) & (common_idx <= backtest_end)
    backtest_dates = common_idx[backtest_mask]
    
    # 初始化预测结果字典，每个因子一个字典
    predictions = {i: {} for i in range(len(f_list))}
    
    for date in backtest_dates: 
        # 获取当天因子数据
        X_date = pd.concat([f_df.loc[date] for f_df in f_list], axis=1)
        X_date.columns = [f'factor_{i}' for i in range(len(f_list))]
        
        # 清理数据
        X_clean = X_date.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(X_clean) > 0:
            # 标准化数据
            X_values = X_clean.values
            
            # 使用Keras模型预测
            pred_factors = model['model'].predict(X_values, verbose=0)
            
            # 将预测结果按因子拆分并存储
            for factor_idx in range(len(f_list)):
                pred_series = pd.Series(pred_factors[:, factor_idx], index=X_clean.index)
                predictions[factor_idx][date] = pred_series
    
    # 将预测结果转换为与f_list结构相同的DataFrame列表
    predicted_factors = []
    for factor_idx in range(len(f_list)):
        factor_df = pd.DataFrame(predictions[factor_idx]).T
        predicted_factors.append(factor_df)
    
    return predicted_factors

def group_analysis(ret_hat, ret, in_sample, group_num=5,draw=True):
    """
    根据预测收益率进行分组回测分析
    返回多空对冲组合的权重DataFrame
    """
    # 对齐数据
    common_dates = ret_hat.index.intersection(ret.index)
    ret_hat_aligned = ret_hat.loc[common_dates]
    ret_aligned = ret.loc[common_dates]
    
    # 定义样本期间
    in_start, in_end = in_sample
    pre_dates = common_dates[common_dates < in_start]
    in_dates = common_dates[(common_dates >= in_start) & (common_dates <= in_end)]
    post_dates = common_dates[common_dates > in_end]
    
    group_returns = pd.DataFrame(index=common_dates)
    # 存储对冲组合权重，明确指定float类型
    hedge_weights = pd.DataFrame(index=common_dates[:-1], columns=ret_hat.columns, dtype=float)
    
    for i, date in enumerate(common_dates[:-1]):
        next_date = common_dates[i + 1]
        pred_returns = ret_hat_aligned.loc[date].dropna()
        
        if len(pred_returns) < group_num:
            continue
            
        try:
            quantiles = pd.qcut(pred_returns, q=group_num, labels=False, duplicates='drop')
            if len(np.unique(quantiles)) < group_num:
                quantiles = pd.qcut(pred_returns.rank(method='first'), q=group_num, labels=False)
            
            next_returns = ret_aligned.loc[next_date]
            
            # 计算对冲组合权重 - 初始化为0.0
            hedge_weight = pd.Series(0.0, index=ret_hat.columns, dtype=float)
            
            for group in range(group_num):
                group_stocks = quantiles[quantiles == group].index
                common_stocks = group_stocks.intersection(next_returns.index)
                
                if len(common_stocks) > 0:
                    group_return = next_returns[common_stocks].mean()
                    group_returns.loc[next_date, f'G{group+1}'] = group_return
                    
                    # 计算对冲组合权重：做多G5，做空G1
                    if group == group_num - 1:  # G5组
                        weight = 1.0 / len(common_stocks)
                        for stock in common_stocks:
                            if stock in hedge_weight.index:
                                hedge_weight[stock] = hedge_weight.get(stock, 0.0) + weight
                    elif group == 0:  # G1组
                        weight = 1.0 / len(common_stocks)
                        for stock in common_stocks:
                            if stock in hedge_weight.index:
                                hedge_weight[stock] = hedge_weight.get(stock, 0.0) - weight
                else:
                    group_returns.loc[next_date, f'G{group+1}'] = 0
            
            # 存储对冲组合权重
            hedge_weights.loc[date] = hedge_weight
            
        except Exception as e:
            continue


    
    group_returns = group_returns.dropna()
    cumulative_returns = (1 + group_returns).cumprod()
    
    # 计算对冲组合
    if 'G1' in group_returns.columns and f'G{group_num}' in group_returns.columns:
        hedge_returns = group_returns[f'G{group_num}'] - group_returns['G1']
        cumulative_hedge = (1 + hedge_returns).cumprod()
    else:
        cumulative_hedge = pd.Series(1.0, index=group_returns.index)

    if not draw:
        return hedge_returns, hedge_weights
    
    # 计算各样本期的Sharpe比率并创建DataFrame
    print("\n" + "="*80)
    print("分组回测Sharpe比率统计")
    print("="*80)
    
    # 定义样本期掩码
    pre_mask = group_returns.index.isin(pre_dates)
    in_mask = group_returns.index.isin(in_dates)
    post_mask = group_returns.index.isin(post_dates)
    
    # 创建DataFrame存储Sharpe比率
    sharpe_data = []
    
    for group in range(group_num):
        group_name = f'G{group+1}'
        if group_name in group_returns.columns:
            returns = group_returns[group_name]
            
            # 各样本期Sharpe
            pre_sharpe = returns[pre_mask].mean() / returns[pre_mask].std() * np.sqrt(12) if len(returns[pre_mask]) > 1 and returns[pre_mask].std() > 0 else 0
            in_sharpe = returns[in_mask].mean() / returns[in_mask].std() * np.sqrt(12) if len(returns[in_mask]) > 1 and returns[in_mask].std() > 0 else 0
            post_sharpe = returns[post_mask].mean() / returns[post_mask].std() * np.sqrt(12) if len(returns[post_mask]) > 1 and returns[post_mask].std() > 0 else 0
            total_sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0
            
            sharpe_data.append({
                'Portfolio': group_name,
                'Pre': pre_sharpe,
                'In': in_sharpe,
                'Post': post_sharpe,
                'Total': total_sharpe
            })
    
    # 对冲组合Sharpe
    if len(hedge_returns) > 0:
        hedge_pre_sharpe = hedge_returns[pre_mask].mean() / hedge_returns[pre_mask].std() * np.sqrt(12) if len(hedge_returns[pre_mask]) > 1 and hedge_returns[pre_mask].std() > 0 else 0
        hedge_in_sharpe = hedge_returns[in_mask].mean() / hedge_returns[in_mask].std() * np.sqrt(12) if len(hedge_returns[in_mask]) > 1 and hedge_returns[in_mask].std() > 0 else 0
        hedge_post_sharpe = hedge_returns[post_mask].mean() / hedge_returns[post_mask].std() * np.sqrt(12) if len(hedge_returns[post_mask]) > 1 and hedge_returns[post_mask].std() > 0 else 0
        hedge_total_sharpe = hedge_returns.mean() / hedge_returns.std() * np.sqrt(12) if hedge_returns.std() > 0 else 0
        
        sharpe_data.append({
            'Portfolio': f'Hedge G{group_num}-G1',
            'Pre': hedge_pre_sharpe,
            'In': hedge_in_sharpe,
            'Post': hedge_post_sharpe,
            'Total': hedge_total_sharpe
        })
    
    # 创建DataFrame并打印
    sharpe_df = pd.DataFrame(sharpe_data)
    sharpe_df.set_index('Portfolio', inplace=True)
    
    # 格式化打印
    print(sharpe_df.to_string(float_format=lambda x: f"{x:6.3f}"))
    print("="*80)
    
    # 绘制组合图（左右轴）
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, group_num))
    for i in range(group_num):
        group_name = f'G{i+1}'
        if group_name in cumulative_returns.columns:
            ax1.plot(cumulative_returns.index, cumulative_returns[group_name],
                    label=f'{group_name}',
                    color=colors[i], linewidth=1.5)
    
    # 右轴显示对冲组合
    ax2 = ax1.twinx()
    ax2.plot(cumulative_hedge.index, cumulative_hedge.values,
             label=f'Hedge G{group_num}-G1',
             color='black', linewidth=2.5, linestyle='--')
    
    # 添加样本期分割竖线
    ax1.axvline(x=pd.to_datetime(in_start), color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=pd.to_datetime(in_end), color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 添加样本期标签
    ax1.text(pd.to_datetime(in_start), 
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.7, 
            'In-sample Start', 
            rotation=90, verticalalignment='bottom', fontsize=10, color='black')

    ax1.text(pd.to_datetime(in_end), 
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.3, 
            'In-sample End', 
            rotation=90, verticalalignment='top', fontsize=10, color='black')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Group Cumulative Return', fontsize=12)
    ax2.set_ylabel('Hedge Cumulative Return', fontsize=12)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.title('Group Returns & Hedge Portfolio Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return hedge_returns, hedge_weights
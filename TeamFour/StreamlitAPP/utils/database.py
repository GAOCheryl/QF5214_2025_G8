import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from io import StringIO
import polars as pl

DB_CONFIGS = {
    'QF5214': {
        'user': 'postgres',
        'password': 'qf5214',
        'host': '134.122.167.14',
        'port': '5555',
    },
}

def get_db_config(connection_name='QF5214'):
    """
    获取指定连接名的数据库配置参数
    
    Args:
        connection_name (str): 连接名
    
    Returns:
        dict: 数据库配置参数
    """
    if connection_name not in DB_CONFIGS:
        raise ValueError(f"未找到指定的连接名: {connection_name}，可用连接名: {list(DB_CONFIGS.keys())}")
    
    return DB_CONFIGS[connection_name]

def parse_db_schema_table(input_str):
    """
    解析输入的字符串，提取数据库名、模式和表名。
    """
    parts = input_str.split('.')
    if len(parts) != 3:
        raise ValueError("输入必须恰好包含两个点作为分隔符，格式为 'dbname.schema.table'")
    return parts[0], parts[1], parts[2]

def read_data(input_str, return_type='pd',connection_name='QF5214', dbname='QF5214'):
    """
    从PostgreSQL数据库加载数据
    
    参数:
    input_str: str, 可以是'dbname.schema.table'格式的表路径，或者是完整的SQL查询语句
    return_type: str, 返回类型 ('pd'或'pl')
    connection_name: str, 连接配置名称
    dbname: str, 默认数据库名，在input_str为SQL查询时使用
    
    返回:
    pandas.DataFrame或polars.DataFrame: 查询结果
    """
    db_config = get_db_config(connection_name)
    
    # 检查input_str是否是SQL查询
    if input_str.strip().upper().startswith('SELECT'):
        # 这是一个SQL查询，使用提供的dbname
        query = input_str
    else:
        # 这是一个表路径格式
        try:
            dbname, schema, table = parse_db_schema_table(input_str)
            # 构建SQL查询字符串
            query = f'SELECT * FROM "{schema}"."{table}";'
        except ValueError:
            raise ValueError("输入必须是SQL查询或'dbname.schema.table'格式")
    
    conn_uri = f'postgresql://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{dbname}'
    
    # 使用read_database_uri函数读取数据
    df = pl.read_database_uri(query=query, uri=conn_uri, engine='adbc')
    
    if return_type == 'pl':
        return df
    else:
        return df.to_pandas()


def write_data(df, table_path, connection_name='QF5214', if_exists='replace'):
    """
    将DataFrame写入PostgreSQL数据库
    
    参数:
    df: pandas.DataFrame或polars.DataFrame, 要写入的数据
    table_path: str, 表路径，格式为'dbname.schema.table'
    connection_name: str, 连接配置名称
    if_exists: str, 如果表已存在，采取的策略 ('replace'或'append')
    
    返回:
    bool: 写入成功返回True，失败返回False
    """
    try:
        # 解析表路径
        dbname, schema, table = parse_db_schema_table(table_path)
        
        # 获取数据库配置
        db_config = get_db_config(connection_name)
        
        # 创建连接URI
        conn_uri = f'postgresql://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{dbname}'
        
        # 创建SQLAlchemy引擎
        engine = create_engine(conn_uri)
        
        # 确保schema存在
        with engine.connect() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            connection.commit()
        
        # 转换为pandas DataFrame（如果是polars DataFrame）
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        # 写入数据
        df.to_sql(
            name=table,
            schema=schema,
            con=engine,
            if_exists=if_exists,
            index=False
        )
        
        print(f"成功写入{len(df)}行数据到{table_path}")
        return True
    
    except Exception as e:
        print(f"写入数据失败: {str(e)}")
        return False


def test_connection():
    """
    测试数据库连接并读取QF5214.tradingstrategy.dailytrading表
    
    返回:
    bool: 连接成功返回True，失败返回False
    pandas.DataFrame: 如果连接成功，返回读取的数据，否则返回None
    """
    try:
        # 尝试连接并读取数据
        df = read_data('QF5214.tradingstrategy.dailytrading', return_type='pd')
        print(f"连接成功！读取到{len(df)}行数据")
        print("数据预览:")
        print(df.head())
        return True, df
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False, None

def get_available_portfolios():
    """
    获取可用的投资组合列表
    
    Returns:
        list: 可用投资组合名称列表
    """
    try:
        # 从portfolio_registry表读取投资组合列表
        portfolio_list = read_data('QF5214.tradingstrategy.portfolio_registry')
        
        # 如果存在Portfolio_Name列，返回该列的唯一值列表
        if 'Portfolio_Name' in portfolio_list.columns:
            return sorted(portfolio_list['Portfolio_Name'].unique().tolist())
        elif 'portfolio_name' in portfolio_list.columns:
            return sorted(portfolio_list['portfolio_name'].unique().tolist())
        else:
            # 如果不存在这些列，返回一个示例列表（临时解决方案）
            return ["Sample_Portfolio_1", "Sample_Portfolio_2", "Sample_Portfolio_3"]
    except Exception as e:
        print(f"获取投资组合列表失败: {str(e)}")
        # 返回一个示例列表作为后备方案
        return ["Sample_Portfolio_1", "Sample_Portfolio_2", "Sample_Portfolio_3"]

def get_portfolio_weights(portfolio_name):
    """
    获取指定投资组合的权重数据
    
    Args:
        portfolio_name (str): 投资组合名称
    
    Returns:
        pandas.DataFrame: 包含日期、股票代码和权重的DataFrame
    """
    try:
        # 查询指定投资组合的权重数据
        query = f"""
        SELECT * FROM "tradingstrategy"."dailytrading" 
        WHERE "Portfolio_Name" = '{portfolio_name}'
        ORDER BY "Date"
        """
        
        weights_df = read_data(query)
        
        # 确保数据有必要的列
        required_cols = ['Date', 'Ticker', 'Weight']
        
        # 处理可能的列名差异
        col_mapping = {
            'Date': ['Date', 'date', 'DATE'],
            'Ticker': ['Ticker', 'ticker', 'code', 'Code', 'TICKER'],
            'Weight': ['Weight', 'weight', 'WEIGHT']
        }
        
        # 对列名进行标准化
        for std_col, possible_cols in col_mapping.items():
            for col in possible_cols:
                if col in weights_df.columns and std_col not in weights_df.columns:
                    weights_df = weights_df.rename(columns={col: std_col})
                    break
        
        # 检查是否有必要的列
        missing_cols = [col for col in required_cols if col not in weights_df.columns]
        if missing_cols:
            print(f"警告：获取的投资组合数据缺少必要的列: {missing_cols}")
            # 如果缺少Weight列，添加一个默认权重（等权重）
            if 'Weight' in missing_cols and 'Ticker' in weights_df.columns:
                unique_dates = weights_df['Date'].unique()
                for date in unique_dates:
                    date_mask = weights_df['Date'] == date
                    stock_count = date_mask.sum()
                    weights_df.loc[date_mask, 'Weight'] = 1.0 / stock_count if stock_count > 0 else 0
        
        # 确保日期列是datetime类型
        if 'Date' in weights_df.columns and weights_df['Date'].dtype != 'datetime64[ns]':
            weights_df['Date'] = pd.to_datetime(weights_df['Date'])
        
        return weights_df
    
    except Exception as e:
        print(f"获取投资组合'{portfolio_name}'的权重数据失败: {str(e)}")
        return None

def get_market_data(start_date=None, end_date=None, tickers=None):
    """
    获取市场数据（股票价格）
    
    Args:
        start_date (str): 开始日期，格式为'YYYY-MM-DD'
        end_date (str): 结束日期，格式为'YYYY-MM-DD'
        tickers (list): 股票代码列表，如果为None则获取所有股票
    
    Returns:
        pandas.DataFrame: 市场数据DataFrame
    """
    try:
        # 构建查询条件
        where_clauses = []
        
        if start_date:
            where_clauses.append(f'"Date" >= \'{start_date}\'')
        
        if end_date:
            where_clauses.append(f'"Date" <= \'{end_date}\'')
        
        if tickers and len(tickers) > 0:
            ticker_list = "', '".join(tickers)
            where_clauses.append(f'"Ticker" IN (\'{ticker_list}\')')
        
        # 组合WHERE子句
        where_sql = " AND ".join(where_clauses)
        where_sql = f"WHERE {where_sql}" if where_sql else ""
        
        # 构建完整查询
        query = f"""
        SELECT * FROM "datacollection"."stock_data" 
        {where_sql}
        ORDER BY "Date", "Ticker"
        """
        
        return read_data(query)
    
    except Exception as e:
        print(f"获取市场数据失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 当直接运行此文件时，执行测试连接
    import polars as pl
    import pandas as pd
    success, data = test_connection()
    if success:
        print("数据库连接测试成功")
    else:
        print("数据库连接测试失败")


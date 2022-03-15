from datetime import datetime, timedelta


__all__ = ['get_days', 'timelevel_re_format']


def get_days(date_str: str) -> str:
    """ 这是一个特殊方法，从一个日期字符串获取一年中的第几天

    Args:
        date_str: 日期字符串 eg:20210927

    Returns:
        eg: 2021137
    """
    d = datetime.strptime(date_str, '%Y%m%d')
    count = d - (datetime(d.year, 1, 1) - timedelta(days=1))

    # 测试时只有137这个文件
    # return '2021137'
    return f'{d.year}{count.days}'


def timelevel_re_format(date_str):
    d = datetime.strptime(date_str, '%Y%m%d%H%M')

    return (d + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M CST')


if __name__ == '__main__':
    print(get_days('20210927'))
    print(timelevel_re_format('202109271910'))

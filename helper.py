def parse_price(val):
    if isinstance(val, str): 
        if "₽" in val:
            val = val.split('₽')[0]
            
        val = val.replace(' ', '')
        return int(val) / 1000000
    
    return float(val)


def parse_area(val):
    if isinstance(val, float): return val
    return float(val.split("м")[0].strip())


def parse_ceiling_height(val):
    if str(val) == 'nan': return -1
    if isinstance(val, float): return val
    if isinstance(val, int): return val
    
    return float(val.split("м")[0])
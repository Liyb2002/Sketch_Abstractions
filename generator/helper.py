import re
import random

def eval_with_range(expr: str, context: dict):
    """
    Evaluate an expression that may contain a range like [0.4, 0.8].
    """
    range_match = re.search(r'\[(.*?)\]', expr)
    if range_match:
        a, b = map(float, range_match.group(1).split(','))
        factor = random.uniform(a, b)
        expr = re.sub(r'\[.*?\]', str(factor), expr)
    return eval(expr, {}, context)

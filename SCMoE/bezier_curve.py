import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

# 计算伯恩斯坦多项式，这是贝塞尔曲线的基础
# 它根据给定的参数i（当前项的索引），n（多项式的阶数），和t（参数，范围在0到1之间），计算并返回多项式的值。
def bernstein_poly(i, n, t):

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

# 定义贝塞尔曲线函数
# 根据给定的控制点集合计算贝塞尔曲线。控制点是一个列表，其中包含一系列的点（每个点是一个列表或元组，包含x和y坐标）
# 比如            [ [1,1],
#                  [2,3],
#                  [4,5], ..[Xn, Yn] ]
# nTimes 参数定义了计算曲线时的时间步数，默认为1000。
def bezier_curve(points, nTimes=1000):

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points]) # x坐标的数组
    yPoints = np.array([p[1] for p in points]) # y坐标的数组

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

# 定义非线性变换函数
def nonlinear_transformation(x, prob=0.5):
    # print(1)
    # print(x)
    if random.random() >= prob:
        return x
    # 官方论文中给定的代码是下面的被注释的一行代码
    # points = [[-1, -1], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    # 下面这是我自己更改的points，我使用官方的代码进行测试发现产生的图片要么是全黑的图片，要么没有变化
    points = [
        [random.uniform(-1.0, 1.0), random.uniform(-0.5, 1.5)],  # 第一个元素范围从-1.0到1.0，第二个从-0.5到1.5
        [random.uniform(-1.0, 1.0), random.uniform(-0.5, 1.5)],
    ]
    xvals, yvals = bezier_curve(points, nTimes=1000000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    nonlinear_x = x*nonlinear_x
    # print(nonlinear_x)
    # print(2)
    return nonlinear_x

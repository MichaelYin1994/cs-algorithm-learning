import pstats
p = pstats.Stats('analysis.out')
p.sort_stats('cumulative').print_stats(50)

# 按照运行时间和函数名进行排序
# p.strip_dirs().sort_stats("cumulative", "name").print_stats(1.5)
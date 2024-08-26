# Class BlockCache
**类属性: 要存什么东西?** 

注意: 要分清楚position value和real position的区别. position value相当于tokens的位置放缩以后的值, 而real position是每个token的实际位置.
1. position value of kv-cache [0, 0.34, 0.91, ... , 72.8]类似这样的
2. blocks 字典 : {N:[a,b]}, 其中 a,b 是某些tokens的实际位置, 即real position


**类方法**


1. update 
import paddle
from hssinfo import cluster

#   //  int nodes     = 6;
#   //  int edges     = 9;
#   //  int rows[]    = {0, 0, 1, 1, 2, 3, 3, 4, 5};
#   //  int cols[]    = {1, 4, 2, 4, 4, 3, 4, 5, 5};
#   //  int weights[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

#   //  int nodes     = 20;
#   //  int edges     = 33;
#   //  int rows[]    = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 7, 9, 10, 10, 11, 11, 11, 12, 12, 15, 15, 16, 16};
#   //  int cols[]    = {15, 17, 16, 17, 7, 8, 15, 11, 13, 15, 9, 13, 14, 19, 13, 15, 17, 19, 18, 12, 14, 11, 12, 17, 16, 17, 19, 13, 15, 16, 19, 17, 19};
#   //  int weights[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

rows = paddle.to_tensor([0, 0, 1, 1, 2, 3, 3, 4, 5], dtype=paddle.int32)
cols = paddle.to_tensor([1, 4, 2, 4, 4, 3, 4, 5, 5], dtype=paddle.int32)
weights = paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=paddle.float32)
nodes = 6
edges = 9

res = cluster(paddle.arange(nodes), rows, cols, weights)
print(res.numpy())

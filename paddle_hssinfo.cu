#include <vector>

#include "paddle/extension.h"
#include "hssinfo.hpp"

std::vector<paddle::Tensor> HSSInfoForward(
    const paddle::Tensor &nodes,
    const paddle::Tensor &rows,
    const paddle::Tensor &cols,
    const paddle::Tensor &weights)
{ // NOLINT
  std::vector<int> rows_vec(rows.numel());
  std::vector<int> cols_vec(cols.numel());
  std::vector<float> weights_vec(weights.numel());

  auto *rows_data = rows.data<int>();
  auto *cols_data = cols.data<int>();
  auto *weights_data = weights.data<float>();
  for (auto i = 0; i < rows.numel(); i++)
  {
    rows_vec[i] = rows_data[i];
    cols_vec[i] = cols_data[i];
    weights_vec[i] = weights_data[i];
  }

  HSSInfo info(nodes.numel(), rows_vec, cols_vec, weights_vec);
  info.CommunityDetection();

  paddle::Tensor out;
  auto *out_data = out.data<int>();
  int out_idx = 0;
  for (auto i = 0; i < info.h_community.size(); i++)
  {
    if (!info.h_community[i].empty())
    {
      for (auto j = 0; j < info.h_community[i].size(); j++, out_idx++)
      {
        out_data[out_idx] = info.h_community[i][j];
      }
    }
  }
  return {out};
}

std::vector<std::vector<int64_t>> HSSInfoInferShape(const std::vector<int64_t> &nodes_shape,
                                                    const std::vector<int64_t> &rows_shape,
                                                    const std::vector<int64_t> &cols_shape,
                                                    const std::vector<int64_t> &weights_shape)
{
  return {nodes_shape};
}

std::vector<paddle::DataType> HSSInfoInferDtype(const paddle::DataType &nodes_type,
                                                const paddle::DataType &rows_type,
                                                const paddle::DataType &cols_type,
                                                const paddle::DataType &weights_type)
{
  return {nodes_type};
}

PD_BUILD_OP(cluster)
    .Inputs({"nodes", "rows", "cols", "weights"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(HSSInfoForward))
    .SetInferShapeFn(PD_INFER_SHAPE(HSSInfoInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HSSInfoInferDtype));

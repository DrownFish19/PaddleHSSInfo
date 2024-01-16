#include <vector>

#include "HSSInfo/hssinfo.cc"
#include "paddle/extension.h"

#include <iostream>

template <typename T>
std::vector<T> Convert2Vector(const paddle::Tensor &tensor) {

  paddle::Tensor tesnor_cpu;
  if (!tensor.is_cpu()) {
    tesnor_cpu = tensor.copy_to(paddle::CPUPlace(), false);
  } else {
    tesnor_cpu = tensor;
  }

  auto numel = tesnor_cpu.numel();
  const T *data_ptr = tesnor_cpu.data<T>();
  std::vector<T> result(data_ptr, data_ptr + numel);
  return result;
}

std::vector<paddle::Tensor> HSSInfoForward(const paddle::Tensor &nodes,
                                           const paddle::Tensor &rows,
                                           const paddle::Tensor &cols,
                                           const paddle::Tensor &weights) {

  std::vector<int> rows_vec = Convert2Vector<int>(rows);
  std::vector<int> cols_vec = Convert2Vector<int>(cols);
  std::vector<float> weights_vec = Convert2Vector<float>(weights);

  HSSInfo info(nodes.numel(), rows_vec, cols_vec, weights_vec);
  info.CommunityDetection();

  paddle::Tensor out = paddle::empty({nodes.numel()}, paddle::DataType::INT32);
  auto *out_data = out.data<int>();
  for (auto i = 0; i < nodes.numel(); i++){
    out_data[i] = -1;
  }
  int group_idx = 0;
  for (auto i = 0; i < info.h_community.size(); i++) {
    if (!info.h_community[i].empty()) {
      for (auto index = info.h_community[i].cbegin(); index < info.h_community[i].cend(); index++) {
        out_data[*index] = group_idx;
      }
      group_idx++;
    }
  }
  return {out};
}

std::vector<std::vector<int64_t>>
HSSInfoInferShape(const std::vector<int64_t> &nodes_shape,
                  const std::vector<int64_t> &rows_shape,
                  const std::vector<int64_t> &cols_shape,
                  const std::vector<int64_t> &weights_shape) {
  return {nodes_shape};
}

std::vector<paddle::DataType> HSSInfoInferDtype(
    const paddle::DataType &nodes_type, const paddle::DataType &rows_type,
    const paddle::DataType &cols_type, const paddle::DataType &weights_type) {
  return {paddle::DataType::INT32};
}

PD_BUILD_OP(cluster)
    .Inputs({"nodes", "rows", "cols", "weights"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(HSSInfoForward))
    .SetInferShapeFn(PD_INFER_SHAPE(HSSInfoInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HSSInfoInferDtype));

// attention_weight_twice_matmul.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>


torch::Tensor attentionWeight_twice_matmul_type1(torch::Tensor weight, torch::Tensor feature) {
    int64_t N = weight.size(0), Q = weight.size(1), K = weight.size(2);
    int64_t C = feature.size(1), FH = feature.size(2), FW = feature.size(3);
    int64_t H = static_cast<int64_t>(std::sqrt(Q)), W = static_cast<int64_t>(std::sqrt(K));
    int64_t scale_H = FH / H, scale_W = FW / W;

    // 4 dimension interpolate
    auto attentionWeight_twice = weight.view({N, H, 1, W, 1, H, 1, W, 1}).repeat({1, 1, scale_H, 1, scale_W, 1, scale_H, 1, scale_W});
    attentionWeight_twice = attentionWeight_twice.view({N, FH * FW, FH * FW});

    feature = feature.view({N, C, FH * FW});
    auto result = torch::einsum("nij, ncj->nci", {attentionWeight_twice, feature});
    result = result.view({N, C, FH, FW});
    return result;
}


torch::Tensor attentionWeight_twice_matmul_type3(torch::Tensor weight, torch::Tensor feature) {
    int64_t N = weight.size(0), Q = weight.size(1), K = weight.size(2);
    int64_t C = feature.size(1), FH = feature.size(2), FW = feature.size(3);
    int64_t H = static_cast<int64_t>(std::sqrt(Q)), W = static_cast<int64_t>(std::sqrt(K));
    int64_t scale_H = FH / H, scale_W = FW / W;

    feature = feature.view({N, C, H, scale_H, W, scale_W}).permute({0, 1, 3, 5, 2, 4}).reshape({N, C, scale_H * scale_W, H * W}).permute({0, 1, 3, 2});
    auto result = torch::einsum("nij, ncjq->nciq", {weight, feature});
    result = result.sum(3);
    result = result.view({N, C, H, 1, W, 1}).repeat({1, 1, 1, scale_H, 1, scale_W}).reshape({N, C, H * scale_H, W * scale_W});
    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attentionWeight_twice_matmul_type1", &attentionWeight_twice_matmul_type1, "Attention Weight Twice Matmul Type1");
    m.def("attentionWeight_twice_matmul_type3", &attentionWeight_twice_matmul_type3, "Attention Weight Twice Matmul Type3");
}
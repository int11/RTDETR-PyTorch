// attention_weight_twice_matmul.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>

torch::Tensor attentionWeight_twice_matmul_type1(torch::Tensor weight, torch::Tensor feature, int scale_factor) {
    auto N = weight.size(0);
    auto Q = weight.size(1);
    auto K = weight.size(2);
    auto QH = static_cast<int>(std::sqrt(Q));
    auto QW = QH;
    auto KH = static_cast<int>(std::sqrt(K));
    auto KW = KH;

    // 4 dimension interpolate
    auto attentionWeight_twice = weight.view({N, QH, 1, QW, 1, KH, 1, KW, 1}).repeat({1, 1, scale_factor, 1, scale_factor, 1, scale_factor, 1, scale_factor});
    attentionWeight_twice = attentionWeight_twice.view({N, Q * scale_factor * scale_factor, K * scale_factor * scale_factor});

    auto Nf = feature.size(0);
    auto C = feature.size(1);
    auto H = feature.size(2);
    auto W = feature.size(3);
    auto result = torch::einsum("nij, ncj->nci", {attentionWeight_twice, feature.view({Nf, C, H * W})});
    result = result.view({Nf, C, H, W});
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attentionWeight_twice_matmul_type1", &attentionWeight_twice_matmul_type1, "Attention Weight Twice Matmul Type1");
}
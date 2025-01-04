#include <torch/extension.h>

namespace monotonic_align {
    torch::Tensor maximum_path_cpp(const torch::Tensor& neg_cent, const torch::Tensor& mask) {
//        auto startk = std::chrono::high_resolution_clock::now();
        torch::NoGradGuard no_grad;
        int64_t batch_size = neg_cent.size(0);
        torch::Tensor path = torch::zeros_like(neg_cent);
        torch::Tensor t_t_max = mask.sum(1).index({torch::indexing::Slice(), 0}).to(torch::kInt);
        torch::Tensor t_s_max = mask.sum(2).index({torch::indexing::Slice(), 0}).to(torch::kInt);
        torch::parallel_for(
            0, batch_size, torch::internal::GRAIN_SIZE,
            [&](int64_t start, int64_t end) {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
                    torch::Tensor path_b = path[batch_idx];
                    torch::Tensor value = neg_cent[batch_idx];
                    int t_y = t_t_max[batch_idx].item<int>();
                    int t_x = t_s_max[batch_idx].item<int>();
                    float max_neg_val=-1000000000.0;
                    int index = t_x - 1;

                    for (int y = 0; y < t_y; y++) {
                        for (int x = std::max(0, t_x + y - t_y); x < std::min(t_x, y + 1); x++) {
                            float v_cur;
                            float v_prev;
                            if (x == y) {
                                v_cur = max_neg_val;
                            }
                            else {
                                v_cur = value.index({y - 1, x}).item<float>();
                            }
                            if (x == 0) {
                                if (y == 0) {
                                    v_prev = 0.;
                                }
                                else {
                                    v_prev = max_neg_val;
                                }
                            }
                            else {
                                v_prev = value.index({y - 1, x - 1}).item<float>();
                            }
                            value.index({y, x}) += std::max(v_prev, v_cur);
                        }
                    }

                    for (int y_n = t_y - 1; y_n >= 0; y_n--) {
                        path_b.index({y_n, index}) = 1;
                        if ((index != 0) && (index == y_n || value.index({y_n - 1, index}).item<float>() < value.index({y_n - 1, index - 1}).item<float>())) {
                            index -= 1;
                        }
                    }
                }
            });
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> duration = end - startk;
//        std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
        return path;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.doc() = "Module with search monotonic alignment algorithm";

    }

    TORCH_LIBRARY(monotonic_align, m) {
        // Note that "float" in the schema corresponds to the C++ double type
        // and the Python float type.
        m.def("maximum_path_cpp(Tensor a, Tensor b) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(monotonic_align, CPU, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }

    TORCH_LIBRARY_IMPL(monotonic_align, CUDA, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }

    TORCH_LIBRARY_IMPL(monotonic_align, HPU, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }

    TORCH_LIBRARY_IMPL(monotonic_align, MPS, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }

    TORCH_LIBRARY_IMPL(monotonic_align, XLA, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }

    TORCH_LIBRARY_IMPL(monotonic_align, IPU, m) {
        m.impl("maximum_path_cpp", &maximum_path_cpp);
    }
}
//
//int main() {
//    torch::NoGradGuard no_grad;
//    torch::Tensor ten = torch::randn({13, 636, 281});
//    torch::Tensor ten2 = torch::randn({13, 636, 281});
//    monotonic_align::maximum_path_cpp(ten, ten2);
//}

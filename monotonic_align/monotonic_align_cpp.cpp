#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>


extern "C" {
    /* Creates a stub empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the at_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__C(void)
    {
        static PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            nullptr,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter states the module,
                       or -1 if the module keeps state in global variables. */
            nullptr,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace monotonic_align {
    at::Tensor maximum_path_cpp(const at::Tensor& neg_cent, const at::Tensor& mask) {
//        auto startk = std::chrono::high_resolution_clock::now();
        int64_t batch_size = neg_cent.size(0);
        at::Tensor path = at::zeros_like(neg_cent);
        at::Tensor t_t_max = mask.sum(1).index({at::indexing::Slice(), 0}).to(at::kInt);
        at::Tensor t_s_max = mask.sum(2).index({at::indexing::Slice(), 0}).to(at::kInt);
        constexpr float max_neg_val = -1000000000.0f;
        at::parallel_for(
            0, batch_size, at::internal::GRAIN_SIZE,
            [&](int64_t start, int64_t end) {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
                    at::Tensor path_b = path[batch_idx];
                    at::Tensor value = neg_cent[batch_idx];
                    int t_y = t_t_max[batch_idx].item<int>();
                    int t_x = t_s_max[batch_idx].item<int>();
                    // float max_neg_val=-1000000000.0;
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
    //
    // at_LIBRARY_IMPL(monotonic_align, HPU, m) {
    //     m.impl("maximum_path_cpp", &maximum_path_cpp);
    // }
    //
    // at_LIBRARY_IMPL(monotonic_align, MPS, m) {
    //     m.impl("maximum_path_cpp", &maximum_path_cpp);
    // }
    //
    // at_LIBRARY_IMPL(monotonic_align, XLA, m) {
    //     m.impl("maximum_path_cpp", &maximum_path_cpp);
    // }
    //
    // at_LIBRARY_IMPL(monotonic_align, IPU, m) {
    //     m.impl("maximum_path_cpp", &maximum_path_cpp);
    // }
}
//
//int main() {
//    at::NoGradGuard no_grad;
//    at::Tensor ten = at::randn({13, 636, 281});
//    at::Tensor ten2 = at::randn({13, 636, 281});
//    monotonic_align::maximum_path_cpp(ten, ten2);
//}

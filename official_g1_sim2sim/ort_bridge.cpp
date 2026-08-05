#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {
thread_local std::string last_error;

struct Runner {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "official_g1_sim2sim"};
  Ort::SessionOptions options;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name;
  std::string output_name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  size_t input_size = 0;
  size_t output_size = 0;

  explicit Runner(const char* model_path) : session(nullptr) {
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(1);
    session = Ort::Session(env, model_path, options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error("expected exactly one model input and one output");
    }
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    input_name = in_name.get();
    output_name = out_name.get();
    input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_size = element_count(input_shape);
    output_size = element_count(output_shape);
  }

  static size_t element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
      if (dim <= 0) throw std::runtime_error("dynamic tensor dimensions are unsupported");
      count *= static_cast<size_t>(dim);
    }
    return count;
  }

  void run(const float* input, size_t input_count, float* output, size_t output_count) {
    if (input_count != input_size || output_count != output_size) {
      throw std::runtime_error("tensor size does not match model metadata");
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input), input_count, input_shape.data(), input_shape.size());
    const char* inputs[] = {input_name.c_str()};
    const char* outputs[] = {output_name.c_str()};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    std::memcpy(output, result[0].GetTensorData<float>(), output_count * sizeof(float));
  }
};

template <typename F>
int guarded(F&& fn) {
  try {
    fn();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}
}  // namespace

extern "C" {
void* ort_runner_create(const char* model_path) {
  try {
    auto runner = std::make_unique<Runner>(model_path);
    last_error.clear();
    return runner.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

void ort_runner_destroy(void* handle) { delete static_cast<Runner*>(handle); }

int ort_runner_input_size(void* handle) {
  return handle ? static_cast<int>(static_cast<Runner*>(handle)->input_size) : -1;
}

int ort_runner_output_size(void* handle) {
  return handle ? static_cast<int>(static_cast<Runner*>(handle)->output_size) : -1;
}

const char* ort_runner_input_name(void* handle) {
  return handle ? static_cast<Runner*>(handle)->input_name.c_str() : "";
}

const char* ort_runner_output_name(void* handle) {
  return handle ? static_cast<Runner*>(handle)->output_name.c_str() : "";
}

int ort_runner_run(void* handle, const float* input, int input_count, float* output,
                   int output_count) {
  if (!handle) {
    last_error = "null runner handle";
    return -1;
  }
  return guarded([&] {
    static_cast<Runner*>(handle)->run(input, static_cast<size_t>(input_count), output,
                                      static_cast<size_t>(output_count));
  });
}

const char* ort_runner_last_error() { return last_error.c_str(); }
}

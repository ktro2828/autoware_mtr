// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/mtr/trt_mtr.hpp"

#include "autoware/mtr/cuda_helper.hpp"
#include "autoware/mtr/intention_point.hpp"
#include "autoware/mtr/trajectory.hpp"
#include "postprocess/postprocess_kernel.cuh"
#include "preprocess/agent_preprocess_kernel.cuh"
#include "preprocess/polyline_preprocess_kernel.cuh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace autoware::mtr
{
TrtMTR::TrtMTR(
  const std::string & model_path, const MTRConfig & config, const BuildConfig & build_config,
  const size_t max_workspace_size)
: config_(config),
  intention_point_(IntentionPoint::from_file(
    config_.num_intention_point_cluster, config_.intention_point_filepath))
{
  max_num_polyline_ = config_.max_num_polyline;
  num_mode_ = config_.num_mode;
  num_future_ = config_.num_future;
  num_intention_point_ = config_.num_intention_point_cluster;

  // build engine
  builder_ = std::make_unique<MTRBuilder>(model_path, build_config, max_workspace_size);
  builder_->setup();

  if (!builder_->isInitialized()) {
    return;
  }

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

bool TrtMTR::doInference(
  const AgentData & agent_data, const PolylineData & polyline_data,
  std::vector<PredictedTrajectory> & trajectories)
{
  initCudaPtr(agent_data, polyline_data);

  if (!preProcess(agent_data, polyline_data)) {
    std::cerr << "Fail to preprocess" << std::endl;
    return false;
  }

  std::vector<void *> buffer = {d_in_trajectory_.get(),      d_in_trajectory_mask_.get(),
                                d_in_polyline_.get(),        d_in_polyline_mask_.get(),
                                d_in_polyline_center_.get(), d_in_last_pos_.get(),
                                d_target_index_.get(),       d_intention_point_.get(),
                                d_out_trajectory_.get(),     d_out_score_.get()};

  if (!builder_->enqueueV2(buffer.data(), stream_, nullptr)) {
    std::cerr << "Fail to do inference" << std::endl;
    return false;
  }

  if (!postProcess(agent_data, trajectories)) {
    std::cerr << "Fail to postprocess" << std::endl;
    return false;
  }

  return true;
}

void TrtMTR::initCudaPtr(const AgentData & agent_data, const PolylineData & polyline_data)
{
  num_target_ = agent_data.num_target();
  num_agent_ = agent_data.num_agent();
  num_timestamp_ = agent_data.time_length();
  num_agent_attr_ = agent_data.input_dim();
  num_polyline_ = polyline_data.num_polyline();
  num_point_ = polyline_data.num_point();
  num_point_attr_ = polyline_data.input_dim();

  // source data
  d_target_index_ = cuda::make_unique<int[]>(num_target_);
  cudaMemset(d_target_index_.get(), 0, num_target_ * sizeof(int));

  d_label_index_ = cuda::make_unique<int[]>(num_agent_);
  cudaMemset(d_label_index_.get(), 0, num_agent_ * sizeof(int));

  d_timestamp_ = cuda::make_unique<float[]>(num_timestamp_);
  cudaMemset(d_timestamp_.get(), 0.0, num_timestamp_ * sizeof(float));

  d_trajectory_ = cuda::make_unique<float[]>(agent_data.size());
  cudaMemset(d_trajectory_.get(), 0.0, agent_data.size() * sizeof(float));

  d_target_state_ = cuda::make_unique<float[]>(num_target_ * agent_data.state_dim());
  cudaMemset(d_target_state_.get(), 0.0, num_target_ * agent_data.state_dim() * sizeof(float));

  d_intention_point_ = cuda::make_unique<float[]>(num_target_ * intention_point_.size());
  cudaMemset(d_intention_point_.get(), 0.0, num_target_ * intention_point_.size() * sizeof(float));

  d_polyline_ = cuda::make_unique<float[]>(polyline_data.size());
  cudaMemset(d_polyline_.get(), 0.0, polyline_data.size() * sizeof(float));

  // preprocessed input
  d_in_trajectory_ =
    cuda::make_unique<float[]>(num_target_ * num_agent_ * num_timestamp_ * num_agent_attr_);
  cudaMemset(
    d_in_trajectory_.get(), 0.0,
    num_target_ * num_agent_ * num_timestamp_ * num_agent_attr_ * sizeof(float));
  d_in_trajectory_mask_ = cuda::make_unique<bool[]>(num_target_ * num_agent_ * num_timestamp_);
  cudaMemset(
    d_in_trajectory_mask_.get(), false, num_target_ * num_agent_ * num_timestamp_ * sizeof(bool));

  d_in_last_pos_ = cuda::make_unique<float[]>(num_target_ * num_agent_ * 3);
  cudaMemset(d_in_last_pos_.get(), 0.0, num_target_ * num_agent_ * 3 * sizeof(float));

  d_in_polyline_ =
    cuda::make_unique<float[]>(num_target_ * max_num_polyline_ * num_point_ * num_point_attr_);
  cudaMemset(
    d_in_polyline_.get(), 0.0,
    num_target_ * max_num_polyline_ * num_point_ * num_point_attr_ * sizeof(float));

  d_in_polyline_mask_ = cuda::make_unique<bool[]>(num_target_ * max_num_polyline_ * num_point_);
  cudaMemset(
    d_in_polyline_.get(), false, num_target_ * max_num_polyline_ * num_point_ * sizeof(bool));

  d_in_polyline_center_ = cuda::make_unique<float[]>(num_target_ * max_num_polyline_ * 3);
  cudaMemset(d_in_polyline_.get(), 0.0, num_target_ * max_num_polyline_ * 3 * sizeof(float));

  if (max_num_polyline_ < num_polyline_) {
    d_tmp_polyline_ =
      cuda::make_unique<float[]>(num_target_ * num_polyline_ * num_point_ * num_point_attr_);
    cudaMemset(
      d_tmp_polyline_.get(), 0.0,
      num_target_ * num_polyline_ * num_point_ * num_point_attr_ * sizeof(float));

    d_tmp_polyline_mask_ = cuda::make_unique<bool[]>(num_target_ * num_polyline_ * num_point_);
    cudaMemset(
      d_tmp_polyline_mask_.get(), false, num_target_ * num_polyline_ * num_point_ * sizeof(bool));

    d_tmp_distance_ = cuda::make_unique<float[]>(num_target_ * num_polyline_);
    cudaMemset(d_tmp_distance_.get(), 0.0, num_target_ * num_polyline_ * sizeof(float));
  }

  // outputs
  d_out_score_ = cuda::make_unique<float[]>(num_target_ * num_mode_);
  cudaMemset(d_out_score_.get(), 0.0, num_target_ * num_mode_ * sizeof(float));

  d_out_trajectory_ =
    cuda::make_unique<float[]>(num_target_ * num_mode_ * num_future_ * PredictedStateDim);
  cudaMemset(
    d_out_trajectory_.get(), 0.0,
    num_target_ * num_mode_ * num_future_ * PredictedStateDim * sizeof(float));

  if (builder_->isDynamic()) {
    // trajectory: (B, N, T, Da)
    builder_->setBindingDimensions(
      0, nvinfer1::Dims4{num_target_, num_agent_, num_timestamp_, num_agent_attr_});
    // trajectory mask: (B, N, T)
    builder_->setBindingDimensions(1, nvinfer1::Dims3{num_target_, num_agent_, num_timestamp_});
    // polyline: (B, K, P, Dp)
    builder_->setBindingDimensions(
      2, nvinfer1::Dims4{num_target_, max_num_polyline_, num_point_, num_point_attr_});
    // polyline mask: (B, K, P)
    builder_->setBindingDimensions(3, nvinfer1::Dims3{num_target_, max_num_polyline_, num_point_});
    // polyline center: (B, K, 3)
    builder_->setBindingDimensions(4, nvinfer1::Dims3{num_target_, max_num_polyline_, 3});
    // agent last position: (B, N, 3)
    builder_->setBindingDimensions(5, nvinfer1::Dims3{num_target_, num_agent_, 3});
    // target indices: (B,)
    nvinfer1::Dims targetIdxDim;
    targetIdxDim.nbDims = 1;
    targetIdxDim.d[0] = num_target_;
    builder_->setBindingDimensions(6, targetIdxDim);
    // intention points: (B, I, 2)
    builder_->setBindingDimensions(7, nvinfer1::Dims3{num_target_, num_intention_point_, 2});
  }
}

bool TrtMTR::preProcess(const AgentData & agent_data, const PolylineData & polyline_data)
{
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_index_.get(), agent_data.target_indices().data(), sizeof(int) * num_target_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_label_index_.get(), agent_data.label_ids().data(), sizeof(int) * num_agent_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_timestamp_.get(), agent_data.timestamps().data(), sizeof(float) * num_timestamp_,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_trajectory_.get(), agent_data.data_ptr(), sizeof(float) * agent_data.size(),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_target_state_.get(), agent_data.target_data_ptr(),
    sizeof(float) * num_target_ * agent_data.state_dim(), cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_polyline_.get(), polyline_data.data_ptr(), sizeof(float) * polyline_data.size(),
    cudaMemcpyHostToDevice, stream_));

  const auto target_label_names = getLabelNames(agent_data.target_label_ids());
  const auto intention_point = intention_point_.as_array(target_label_names);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    d_intention_point_.get(), intention_point.data(),
    sizeof(float) * num_target_ * intention_point_.size(), cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(agentPreprocessLauncher(
    num_target_, num_agent_, num_timestamp_, agent_data.state_dim(), agent_data.num_class(),
    agent_data.ego_index(), d_target_index_.get(), d_label_index_.get(), d_timestamp_.get(),
    d_trajectory_.get(), d_in_trajectory_.get(), d_in_trajectory_mask_.get(), d_in_last_pos_.get(),
    stream_));

  auto T = 11;
  // auto D = 12;
  // auto C = 3;

  auto B = 2;
  auto N = 2;
  {
    std::vector<float> host_buffer(num_target_ * num_agent_ * 3);
    cudaMemcpy(
      host_buffer.data(), d_in_last_pos_.get(), num_target_ * num_agent_ * 3 * sizeof(float),
      cudaMemcpyDeviceToHost);
    std::cerr << "Preprocessed lat position \n";
    std::vector<std::string> values{"x", "y", "z"};
    for (int b = 0; b < B; b++) {
      for (int n = 0; n < N; n++) {
        std::cerr << "{b: " << b;
        std::cerr << ",n: " << n << ": ";
        for (int i = 0; i < 3; ++i) {
          std::cerr << values[i] << ": " << host_buffer[b * N * 3 + n * 3 + i] << ",";
        }
        std::cerr << "}\n";
      }
    }
  }

  {
    std::vector<float> host_buffer(N * B * T * 29);

    // Step 2: Copy data from GPU to host
    cudaMemcpy(
      host_buffer.data(), d_in_trajectory_.get(), N * B * T * 29 * sizeof(float),
      cudaMemcpyDeviceToHost);

    std::cerr << "Preprocessed output \n";
    std::vector<std::string> values{"x",    "y",  "z",  "L",  "W",  "H",   "O0",  "O1",
                                    "O2",   "O3", "O4", "T0", "T1", "T2",  "T3",  "T4",
                                    "T5",   "T6", "T7", "T8", "T9", "T10", "T11", "Yaw0",
                                    "Yaw1", "Vx", "Vy", "Ax", "Ay"};
    for (int b = 0; b < B; b++) {
      for (int n = 0; n < N; n++) {
        std::cerr << "{b: " << b;
        std::cerr << ",n: " << n << ": \n";
        for (int t = 0; t < T; ++t) {
          for (int i = 0; i < 29; ++i) {
            std::cerr << values[i] << ": " << host_buffer[b * N * T * 29 + (n * T + t) * 29 + i]
                      << ",";
          }
          std::cerr << "\n";
        }
        std::cerr << "}\n";
      }
    }
  }
  if (max_num_polyline_ < num_polyline_) {
    CHECK_CUDA_ERROR(polylinePreprocessWithTopkLauncher(
      max_num_polyline_, num_polyline_, num_point_, polyline_data.state_dim(), d_polyline_.get(),
      num_target_, agent_data.state_dim(), d_target_state_.get(), d_tmp_polyline_.get(),
      d_tmp_polyline_mask_.get(), d_tmp_distance_.get(), d_in_polyline_.get(),
      d_in_polyline_mask_.get(), d_in_polyline_center_.get(), stream_));
  } else {
    CHECK_CUDA_ERROR(polylinePreprocessLauncher(
      num_polyline_, num_point_, polyline_data.state_dim(), d_polyline_.get(), num_target_,
      agent_data.state_dim(), d_target_state_.get(), d_in_polyline_.get(),
      d_in_polyline_mask_.get(), d_in_polyline_center_.get(), stream_));
  }

  // // Check for NaN or invalid values in d_in_polyline_center_
  {
    std::vector<float> host_buffer(num_target_ * max_num_polyline_ * 3);
    cudaMemcpy(
      host_buffer.data(), d_in_polyline_center_.get(),
      num_target_ * max_num_polyline_ * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    for (const auto & val : host_buffer) {
      if (std::isnan(val) || std::abs(val) > 1000) {
        std::cerr << "NaN found in d_in_polyline_center_" << std::endl;
        if (!std::isnan(val)) {
          std::cerr << "high value " << val << std::endl;
        }
      }
    }
  }

  // Check for NaN or invalid values in d_in_polyline_
  {
    std::vector<float> host_buffer(num_target_ * max_num_polyline_ * num_point_ * num_point_attr_);
    cudaMemcpy(
      host_buffer.data(), d_in_polyline_.get(),
      num_target_ * max_num_polyline_ * num_point_ * num_point_attr_ * sizeof(float),
      cudaMemcpyDeviceToHost);
    std::cerr << "polyline data has "
              << num_target_ * max_num_polyline_ * num_point_ * num_point_attr_ << " elements\n";
    size_t count = 0;
    for (const auto & val : host_buffer) {
      if (std::isnan(val)) {
        std::cerr << "NaN found in d_in_polyline_ for element" << count++ << std::endl;
      }
    }
  }

  // Check for NaN or invalid values in d_intention_point_
  {
    std::vector<float> host_buffer(num_target_ * intention_point_.size());
    cudaMemcpy(
      host_buffer.data(), d_intention_point_.get(),
      num_target_ * intention_point_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (const auto & val : host_buffer) {
      if (std::isnan(val)) {
        std::cerr << "NaN found in d_intention_point_" << std::endl;
      }
    }
  }
  return true;
}

bool TrtMTR::postProcess(
  const AgentData & agent_data, std::vector<PredictedTrajectory> & trajectories)
{
  CHECK_CUDA_ERROR(postprocessLauncher(
    num_target_, num_mode_, num_future_, agent_data.state_dim(), d_target_state_.get(),
    PredictedStateDim, d_out_trajectory_.get(), stream_));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  h_out_score_.clear();
  h_out_trajectory_.clear();
  h_out_score_.resize(num_target_ * num_mode_);
  h_out_trajectory_.resize(num_target_ * num_mode_ * num_future_ * PredictedStateDim);

  CHECK_CUDA_ERROR(cudaMemcpy(
    h_out_score_.data(), d_out_score_.get(), sizeof(float) * num_target_ * num_mode_,
    cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(
    h_out_trajectory_.data(), d_out_trajectory_.get(),
    sizeof(float) * num_target_ * num_mode_ * num_future_ * PredictedStateDim,
    cudaMemcpyDeviceToHost));

  trajectories.clear();
  trajectories.reserve(num_target_);
  std::vector<std::string> values{"x", "y", "xmean", "ymean", "std_dev", "vx", "vy"};

  for (auto b = 0; b < num_target_; ++b) {
    const auto score_itr = h_out_score_.cbegin() + b * num_mode_;
    const std::vector<double> scores(score_itr, score_itr + num_mode_);
    const auto mode_itr =
      h_out_trajectory_.cbegin() + b * num_mode_ * num_future_ * PredictedStateDim;
    std::vector<double> modes(mode_itr, mode_itr + num_mode_ * num_future_ * PredictedStateDim);
    std::cerr << "Target " << b << "\n";
    for (size_t i = 0; i < static_cast<size_t>(num_mode_); i++) {
      std::cerr << "Mode " << i << "\n";
      for (size_t j = 0; j < static_cast<size_t>(num_future_); j++) {
        std::cerr << "Step " << j << "\n";
        for (size_t k = 0; k < PredictedStateDim; k++) {
          std::cerr << values[k] << ": "
                    << modes[i * num_future_ * PredictedStateDim + j * PredictedStateDim + k]
                    << ",";
        }
        std::cerr << "\n";
      }
    }
    trajectories.emplace_back(scores, modes, num_mode_, num_future_);
  }
  return true;
}
}  // namespace autoware::mtr

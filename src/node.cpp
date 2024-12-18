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

#include "autoware/mtr/node.hpp"

#include "autoware/mtr/agent.hpp"
#include "autoware/mtr/map_conversion.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <rclcpp/logging.hpp>

#include <autoware_perception_msgs/msg/detail/tracked_object__struct.hpp>
#include <geometry_msgs/msg/detail/point__builder.hpp>
#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <geometry_msgs/msg/detail/twist__struct.hpp>
#include <geometry_msgs/msg/detail/twist_with_covariance__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace autoware::mtr
{
namespace
{
// Convert `TrackedObject` to `AgentState`.
AgentState to_agent_state(const TrackedObject & object, const bool is_valid)
{
  const auto & pose = object.kinematics.pose_with_covariance.pose;
  const auto & twist = object.kinematics.twist_with_covariance.twist;
  const auto & accel = object.kinematics.acceleration_with_covariance.accel;
  const auto & dimensions = object.shape.dimensions;

  const float yaw = tf2::getYaw(pose.orientation);
  const float valid = is_valid ? 1.0f : 0.0f;

  return {pose.position, dimensions, yaw, twist.linear, accel.linear, valid};
}

// Get the label index corresponding to AgentLabel. If the label of tracked object is not * defined
// in AgentLabel returns `-1`.
int to_label_id(const TrackedObject & object)
{
  const auto classification =
    autoware::object_recognition_utils::getHighestProbLabel(object.classification);
  if (autoware::object_recognition_utils::isCarLikeVehicle(classification)) {
    return AgentLabel::VEHICLE;
  }
  if (classification == ObjectClassification::PEDESTRIAN) {
    return AgentLabel::PEDESTRIAN;
  }
  if (
    classification == ObjectClassification::MOTORCYCLE ||
    classification == ObjectClassification::BICYCLE) {
    return AgentLabel::CYCLIST;
  }
  return -1;  // other labels
}
}  // namespace

MTRNode::MTRNode(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("mtr", node_options), transform_listener_(this), ego_states_(100), timestamps_(11)
{
  // Setup MTR
  {
    // Model config
    const auto model_path = declare_parameter<std::string>("model_params.model_path");
    const auto target_labels =
      declare_parameter<std::vector<std::string>>("model_params.target_labels");
    const auto num_past = static_cast<size_t>(declare_parameter<int>("model_params.num_past"));
    const auto num_mode = static_cast<size_t>(declare_parameter<int>("model_params.num_mode"));
    const auto num_future = static_cast<size_t>(declare_parameter<int>("model_params.num_future"));
    const auto max_num_polyline =
      static_cast<size_t>(declare_parameter<int>("model_params.max_num_polyline"));
    const auto max_num_point =
      static_cast<size_t>(declare_parameter<int>("model_params.max_num_point"));
    const auto point_break_distance =
      static_cast<float>(declare_parameter<double>("model_params.point_break_distance"));
    const auto intention_point_filepath =
      declare_parameter<std::string>("model_params.intention_point_filepath");
    const auto num_intention_point_cluster =
      static_cast<size_t>(declare_parameter<int>("model_params.num_intention_point_cluster"));
    config_ptr_ = std::make_unique<MTRConfig>(
      target_labels, num_past, num_mode, num_future, max_num_polyline, max_num_point,
      point_break_distance, intention_point_filepath, num_intention_point_cluster);
    // Build config
    const auto is_dynamic = declare_parameter<bool>("build_params.is_dynamic");
    const auto precision = declare_parameter<std::string>("build_params.precision");
    const auto calibration = declare_parameter<std::string>("build_params.calibration");
    build_config_ptr_ = std::make_unique<BuildConfig>(is_dynamic, precision, calibration);
    model_ptr_ = std::make_unique<TrtMTR>(model_path, *config_ptr_, *build_config_ptr_);
  }

  // Setup node
  sub_objects_ = create_subscription<TrackedObjects>(
    "~/input/objects", rclcpp::QoS{1}, std::bind(&MTRNode::callback, this, std::placeholders::_1));
  sub_map_ = create_subscription<LaneletMapBin>(
    "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&MTRNode::onMap, this, std::placeholders::_1));

  pub_objects_ = create_publisher<PredictedObjects>("~/output/objects", rclcpp::QoS{1});

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

TrackedObject MTRNode::makeEgoTrackedObject(const Odometry::ConstSharedPtr ego_msg) const
{
  auto createPoint32 =
    [](const double x, const double y, const double z) -> geometry_msgs::msg::Point32 {
    geometry_msgs::msg::Point32 p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
  };

  TrackedObject output;
  const auto ego_pose = ego_msg->pose;
  const auto twist = ego_msg->twist;

  // Classification and probability
  {
    output.existence_probability = 1.0;
    ObjectClassification classification;
    classification.label = ObjectClassification::CAR;
    output.classification = {classification};
  }

  // Kinematics
  {
    output.kinematics.pose_with_covariance = ego_pose;
    output.kinematics.twist_with_covariance = twist;
  }
  // Shape
  {
    const auto & ego_max_long_offset = vehicle_info_.max_longitudinal_offset_m;
    const auto & ego_rear_overhang = vehicle_info_.vehicle_height_m;
    const auto & ego_length = vehicle_info_.vehicle_length_m;
    const auto & ego_width = vehicle_info_.vehicle_width_m;
    const auto & ego_height = vehicle_info_.vehicle_height_m;

    autoware_perception_msgs::msg::Shape shape;
    shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    shape.dimensions.x = ego_length;
    shape.dimensions.y = ego_width;
    shape.dimensions.z = ego_height;

    // TODO(Daniel): Should use overhang and ego info utils
    geometry_msgs::msg::Point32 p;
    shape.footprint.points.push_back(
      createPoint32(-ego_rear_overhang, -ego_width / 2.0, ego_height));
    shape.footprint.points.push_back(
      createPoint32(-ego_rear_overhang, ego_width / 2.0, ego_height));
    shape.footprint.points.push_back(
      createPoint32(ego_max_long_offset, ego_width / 2.0, ego_height));
    shape.footprint.points.push_back(
      createPoint32(ego_max_long_offset, -ego_width / 2.0, ego_height));
    output.shape = shape;
  }
  return TrackedObject(output);
}

template <typename T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & values)
{
  os << "(";
  for (const auto & v : values) {
    os << v << " ";
  }
  os << ")";
  return os;
}

void MTRNode::callback(const TrackedObjects::ConstSharedPtr object_msg)
{
  const auto current_ego_msg = fetchEgoState();
  if (!current_ego_msg) {
    RCLCPP_WARN(get_logger(), "No ego data");
    return;
  }
  if (!polyline_ptr_) {
    RCLCPP_WARN(get_logger(), "No polyline");
    return;
  }

  const auto current_time = static_cast<float>(rclcpp::Time(object_msg->header.stamp).seconds());

  timestamps_.push_back(current_time);

  removeAncientHistory(current_time, object_msg);
  updateHistory(current_time, object_msg, current_ego_msg.value());

  std::vector<std::string> object_ids;
  std::vector<AgentHistory> histories;
  std::vector<size_t> label_indices;
  histories.reserve(agent_history_map_.size());
  object_ids.reserve(agent_history_map_.size());
  label_indices.reserve(agent_history_map_.size());
  int sdc_index = -1;
  for (const auto & [object_id, history] : agent_history_map_) {
    object_ids.emplace_back(object_id);
    histories.emplace_back(history);
    label_indices.emplace_back(history.label_index());
    if (object_id == EGO_ID) {
      sdc_index = histories.size() - 1;
    }
  }

  if (sdc_index == -1) {
    RCLCPP_WARN(get_logger(), "No EGO");
    return;
  }

  const auto target_indices = extractTarget(histories, object_msg->header);
  if (target_indices.empty()) {
    RCLCPP_WARN(get_logger(), "No target agents");
    return;
  }

  const auto relative_timestamps = getRelativeTimestamps();
  AgentData agent_data(
    histories, static_cast<size_t>(sdc_index), target_indices, label_indices, relative_timestamps);

  std::vector<PredictedTrajectory> trajectories;
  if (!model_ptr_->doInference(agent_data, *polyline_ptr_, trajectories)) {
    RCLCPP_WARN(get_logger(), "Inference failed");
    return;
  }

  PredictedObjects output;
  output.header = object_msg->header;
  output.objects.reserve(target_indices.size());
  for (size_t i = 0; i < target_indices.size(); ++i) {
    const auto & target_idx = target_indices.at(i);
    const auto & object_id = object_ids.at(target_idx);
    const auto & object = object_msg_map_.at(object_id);
    const auto & trajectory = trajectories.at(i);

    auto predicted_object = generatePredictedObject(object, trajectory);
    output.objects.emplace_back(predicted_object);
  }

  // Publish results
  pub_objects_->publish(output);
}

void MTRNode::onMap(const LaneletMapBin::ConstSharedPtr map_msg)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *map_msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);

  RCLCPP_DEBUG(get_logger(), "Start converting lanelet to polyline");
  polyline_ptr_ = lanelet_to_polyline(
    lanelet_map_ptr_, config_ptr_->max_num_polyline, config_ptr_->max_num_point,
    config_ptr_->point_break_distance);
  if (polyline_ptr_) {
    RCLCPP_DEBUG(get_logger(), "Success to convert lanelet to polyline");
  } else {
    RCLCPP_WARN(get_logger(), "Fail to convert lanelet to polyline");
  }
}

std::optional<TrackedObject> MTRNode::fetchEgoState()
{
  const Odometry::ConstSharedPtr ego_msg = sub_ego_.takeData();
  if (!ego_msg) {
    return std::nullopt;
  }

  const auto current_time = static_cast<float>(rclcpp::Time(ego_msg->header.stamp).seconds());
  const auto & position = ego_msg->pose.pose.position;
  const auto & twist = ego_msg->twist.twist;

  const float yaw = tf2::getYaw(ego_msg->pose.pose.orientation);

  float ax = 0.0f;
  float ay = 0.0f;
  const auto & latest_state = *ego_states_.end();
  const auto time_diff = current_time - latest_state.first;
  ax = (static_cast<float>(twist.linear.x) - latest_state.second.vx()) / (time_diff + 1e-10f);
  ay = (static_cast<float>(twist.linear.y) - latest_state.second.vy()) / (time_diff + 1e-10f);

  const auto acceleration = geometry_msgs::build<geometry_msgs::msg::Vector3>().x(ax).y(ay).z(0.0);

  const auto dimensions = geometry_msgs::build<geometry_msgs::msg::Vector3>()
                            .x(vehicle_info_.vehicle_length_m)
                            .y(vehicle_info_.vehicle_width_m)
                            .z(vehicle_info_.vehicle_height_m);

  ego_states_.push_back(std::make_pair(
    current_time, AgentState(position, dimensions, yaw, twist.linear, acceleration, 1.0f)));

  // make the ego vehicle a tracked object
  return makeEgoTrackedObject(ego_msg);
}

void MTRNode::removeAncientHistory(
  const float current_time, const TrackedObjects::ConstSharedPtr objects_msg)
{
  constexpr float time_threshold = 1.0f;  // TODO(ktro2828): use parameter
  for (const auto & object : objects_msg->objects) {
    const auto & object_id = autoware::universe_utils::toHexString(object.object_id);
    if (agent_history_map_.count(object_id) == 0) {
      continue;
    }

    const auto & history = agent_history_map_.at(object_id);
    if (history.is_ancient(current_time, time_threshold)) {
      agent_history_map_.erase(object_id);
    }
  }

  if (
    agent_history_map_.count(EGO_ID) != 0 &&
    agent_history_map_.at(EGO_ID).is_ancient(current_time, time_threshold)) {
    agent_history_map_.erase(EGO_ID);
  }
}

void MTRNode::updateHistory(
  const float current_time, const TrackedObjects::ConstSharedPtr objects_msg,
  const TrackedObject current_ego_msg)
{
  std::vector<std::string> observed_ids;
  // Other agents
  for (const auto & object : objects_msg->objects) {
    auto label_id = to_label_id(object);
    if (label_id == -1) {
      continue;
    }

    const auto & object_id = autoware::universe_utils::toHexString(object.object_id);
    observed_ids.emplace_back(object_id);
    if (object_msg_map_.count(object_id) == 0) {
      object_msg_map_.emplace(object_id, object);
    } else {
      object_msg_map_.at(object_id) = object;
    }

    AgentState state = to_agent_state(object, true);
    if (agent_history_map_.count(object_id) == 0) {
      AgentHistory history(object_id, label_id, config_ptr_->num_past);
      history.update(current_time, state);
      agent_history_map_.emplace(object_id, history);
    } else {
      agent_history_map_.at(object_id).update(current_time, state);
    }
  }

  // Ego vehicle
  observed_ids.emplace_back(EGO_ID);
  if (object_msg_map_.count(EGO_ID) == 0) {
    object_msg_map_.emplace(EGO_ID, current_ego_msg);
  } else {
    object_msg_map_.at(EGO_ID) = current_ego_msg;
  }

  AgentState ego_state = currentEgoState(current_time);
  if (agent_history_map_.count(EGO_ID) == 0) {
    AgentHistory history(EGO_ID, AgentLabel::VEHICLE, config_ptr_->num_past);
    history.update(current_time, ego_state);
    agent_history_map_.emplace(EGO_ID, history);
  } else {
    agent_history_map_.at(EGO_ID).update(current_time, ego_state);
  }

  // update unobserved histories with empty
  for (auto & [object_id, history] : agent_history_map_) {
    if (std::find(observed_ids.cbegin(), observed_ids.cend(), object_id) != observed_ids.cend()) {
      continue;
    }
    history.update_empty();
  }
}

AgentState MTRNode::currentEgoState(const float current_time) const
{
  auto state =
    std::min_element(ego_states_.begin(), ego_states_.end(), [&](const auto & s1, const auto & s2) {
      return std::abs(s1.first - current_time) < std::abs(s2.first - current_time);
    });
  return state->second;
}

std::vector<size_t> MTRNode::extractTarget(
  const std::vector<AgentHistory> & histories, const std_msgs::msg::Header & header)
{
  const auto map2ego = transform_listener_.getTransform(
    "base_link", header.frame_id, header.stamp, rclcpp::Duration::from_seconds(0.1));

  if (!map2ego) {
    RCLCPP_WARN(get_logger(), "Failed to get transform from map to base_link.");
    return {};
  }

  std::vector<std::pair<size_t, double>> distances;
  for (size_t i = 0; i < histories.size(); ++i) {
    const auto & history = histories.at(i);
    if (!history.is_valid_latest() || history.object_id() == EGO_ID) {
      // TODO(ktro2828): allow to push ego state
      distances.emplace_back(i, INFINITY);
    } else {
      const auto state = history.get_latest_state();
      geometry_msgs::msg::PoseStamped pose_in_map;
      pose_in_map.header = header;
      pose_in_map.pose.position.x = state.x();
      pose_in_map.pose.position.y = state.y();
      pose_in_map.pose.position.z = state.z();
      pose_in_map.pose.orientation = autoware::universe_utils::createQuaternionFromYaw(state.yaw());

      geometry_msgs::msg::PoseStamped pose_in_ego;
      tf2::doTransform(pose_in_map, pose_in_ego, *map2ego);

      const auto dist = std::hypot(
        pose_in_ego.pose.position.x, pose_in_ego.pose.position.y, pose_in_ego.pose.position.z);
      distances.emplace_back(i, dist);
    }
  }

  std::sort(distances.begin(), distances.end(), [](const auto & item1, const auto & item2) {
    return item1.second < item2.second;
  });

  // TODO(ktro2828): use a parameter
  constexpr size_t max_target_size = 1;
  constexpr double distance_threshold = 1000.0;

  std::vector<size_t> target_indices;
  target_indices.reserve(max_target_size);
  for (const auto & [idx, value] : distances) {
    if (value < distance_threshold) {
      target_indices.emplace_back(idx);
    }
    if (max_target_size <= target_indices.size()) {
      break;
    }
  }

  return target_indices;
}

std::vector<float> MTRNode::getRelativeTimestamps() const
{
  std::vector<float> output;
  output.reserve(timestamps_.size());
  for (const auto & t : timestamps_) {
    output.push_back(t - *timestamps_.begin());
  }
  return output;
}

PredictedObject MTRNode::generatePredictedObject(
  const TrackedObject & object, const PredictedTrajectory & trajectory)
{
  const auto & init_pose_with_cov = object.kinematics.pose_with_covariance;
  const auto & init_twist_with_cov = object.kinematics.twist_with_covariance;
  const auto & init_accel_with_cov = object.kinematics.acceleration_with_covariance;

  PredictedObject predicted_object;
  predicted_object.kinematics.initial_pose_with_covariance = init_pose_with_cov;
  predicted_object.kinematics.initial_twist_with_covariance = init_twist_with_cov;
  predicted_object.kinematics.initial_acceleration_with_covariance = init_accel_with_cov;
  predicted_object.classification = object.classification;
  predicted_object.shape = object.shape;
  predicted_object.object_id = object.object_id;

  float max_existence_probability = 0.0f;
  for (const auto & mode : trajectory.get_modes()) {
    PredictedPath waypoints;
    waypoints.confidence = mode.score();
    waypoints.time_step = rclcpp::Duration::from_seconds(0.1);  // TODO(ktro282): use a parameter
    waypoints.path.reserve(mode.num_future());
    if (max_existence_probability < mode.score()) {
      max_existence_probability = mode.score();
    }

    for (const auto & state : mode.get_waypoints()) {
      geometry_msgs::msg::Pose predicted_pose;
      predicted_pose.position.x = static_cast<double>(state.x());
      predicted_pose.position.y = static_cast<double>(state.y());
      predicted_pose.position.z = init_pose_with_cov.pose.position.z;
      predicted_pose.orientation = init_pose_with_cov.pose.orientation;
      waypoints.path.emplace_back(predicted_pose);
      if (waypoints.path.size() >= waypoints.path.max_size()) {
        break;
      }
    }
    predicted_object.kinematics.predicted_paths.emplace_back(waypoints);
  }
  predicted_object.existence_probability = max_existence_probability;

  return predicted_object;
}
}  // namespace autoware::mtr

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::mtr::MTRNode);

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

#ifndef AUTOWARE__MTR__MAP_CONVERSION_HPP_
#define AUTOWARE__MTR__MAP_CONVERSION_HPP_

#include "polyline.hpp"

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/CompoundPolygon.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/utility/Optional.h>
#include <lanelet2_routing/Forward.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace autoware::mtr
{
/**
 * @brief Convert lanelet map to polylines.
 *
 * @param lanelet_map_ptr
 * @return std::shared_ptr<PolylineData>
 */
std::shared_ptr<PolylineData> lanelet_to_polyline(
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr, size_t max_num_polyline,
  size_t max_num_point, float point_break_distance);

/**
 * @brief Convert 3D linestring to lane points.
 *
 * @param linestring
 * @return std::vector<LanePoint>
 */
std::vector<LanePoint> from_linestring(const lanelet::ConstLineString3d & linestring);

/**
 * @brief Convert 3D polygon to lane points.
 *
 * @param polygon
 * @return std::vector<LanePoint>
 */
std::vector<LanePoint> from_polygon(const lanelet::CompoundPolygon3d & polygon);

/**
 * @brief Insert lane points into the container from the end of it.
 *
 * @param points Sequence of points to be inserted.
 * @param container Points container.
 */
inline void insert_lane_points(
  const std::vector<LanePoint> & points, std::vector<LanePoint> & container)
{
  container.reserve(container.size() * 2);
  container.insert(container.end(), points.begin(), points.end());
}

/**
 * @brief Extract the type name from a lanelet.
 *
 * @param lanelet
 * @return lanelet::Optional<std::string>
 */
inline lanelet::Optional<std::string> to_type_name(const lanelet::Lanelet & lanelet) noexcept
{
  return lanelet.hasAttribute("type") ? lanelet.attribute("type").as<std::string>()
                                      : lanelet::Optional<std::string>();
}

/**
 * @brief Extract the type name from a 3D linestring.
 *
 * @param linestring
 * @return lanelet::Optional<std::string>
 */
inline lanelet::Optional<std::string> to_type_name(
  const lanelet::ConstLineString3d & linestring) noexcept
{
  return linestring.hasAttribute("type") ? linestring.attribute("type").as<std::string>()
                                         : lanelet::Optional<std::string>();
}

/**
 * @brief Extract the subtype name from a lanelet.
 *
 * @param lanelet
 * @return std::optional<string>
 */
inline lanelet::Optional<std::string> to_subtype_name(const lanelet::Lanelet & lanelet) noexcept
{
  return lanelet.hasAttribute("subtype") ? lanelet.attribute("subtype").as<std::string>()
                                         : lanelet::Optional<std::string>();
}

/**
 * @brief Extract the subtype name from a 3D linestring.
 *
 * @param linestring
 * @return lanelet::Optional<std::string>
 */
inline lanelet::Optional<std::string> to_subtype_name(
  const lanelet::ConstLineString3d & linestring) noexcept
{
  return linestring.hasAttribute("subtype") ? linestring.attribute("subtype").as<std::string>()
                                            : lanelet::Optional<std::string>();
}

/**
 * @brief Check if the specified lanelet is the turnable intersection.
 *
 * @param lanelet
 * @return true
 * @return false
 */
inline bool is_turnable_intersection(const lanelet::Lanelet & lanelet)
{
  return lanelet.hasAttribute("turn_direction");
}

/**
 * @brief Check if the specified lanelet subtype is kind of lane.
 *
 * @param subtype
 * @return True if the lanelet subtype is the one of the (road, highway, road_shoulder,
 * pedestrian_lane, bicycle_lane, walkway).
 */
inline bool is_lane_like(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }
  const auto & subtype_str = subtype.value();
  return (
    subtype_str == "road" || subtype_str == "highway" || subtype_str == "road_shoulder" ||
    subtype_str == "pedestrian_lane" || subtype_str == "bicycle_lane" || subtype_str == "walkway");
}

/**
 * @brief Check if the specified lanelet subtype is kind of the roadway.
 *
 * @param subtype Subtype of the corresponding lanelet.
 * @return True if the subtype is the one of the (road, highway, road_shoulder).
 */
inline bool is_roadway_like(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }
  const auto & subtype_str = subtype.value();
  return subtype_str == "road" || subtype_str == "highway" || subtype_str == "road_shoulder";
}

/**
 * @brief Check if the specified linestring is kind of the boundary.
 *
 * @param linestring 3D linestring.
 * @return True if the type is the one of the (line_thin, line_thick, road_boarder) and the subtype
 * is not virtual.
 */
inline bool is_boundary_like(const lanelet::ConstLineString3d & linestring)
{
  const auto type = to_type_name(linestring);
  const auto subtype = to_subtype_name(linestring);
  if (!type || !subtype) {
    return false;
  }

  const auto & type_str = type.value();
  const auto & subtype_str = subtype.value();
  return (type_str == "line_thin" || type_str == "line_thick" || type_str == "road_boarder") &&
         subtype_str != "virtual";
}

/**
 * @brief Check if the specified linestring is the kind of crosswalk.
 *
 * @param subtype Subtype of the corresponding polygon.
 * @return True if the lanelet subtype is the one of the (crosswalk,).
 */
inline bool is_crosswalk_like(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }

  const auto & subtype_str = subtype.value();
  return subtype_str == "crosswalk";
}

}  // namespace autoware::mtr

#endif  // AUTOWARE__MTR__MAP_CONVERSION_HPP_

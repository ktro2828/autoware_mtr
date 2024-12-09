# autoware_mtr

## Purpose

The `autoware_mtr` package is used for 3D object motion prediction based on ML-based model called MTR.

## Inner-workings / Algorithms

The implementation bases on MTR [1] work. It uses TensorRT library for data process and network interface.

## Inputs / Outputs

### Input

| Name                 | Type                                            | Description              |
| -------------------- | ----------------------------------------------- | ------------------------ |
| `~/input/objects`    | `autoware_perception_msgs::msg::TrackedObjects` | Input agent state.       |
| `~/input/vector_map` | `autoware_map_msgs::msg::LeneletMapBin`         | Input vector map.        |
| `~/input/ego`        | `sensor_msgs::msg::Odometry`                    | Input ego vehicle state. |

### Output

| Name               | Type                                              | Description                |
| ------------------ | ------------------------------------------------- | -------------------------- |
| `~/output/objects` | `autoware_perception_msgs::msg::PredictedObjects` | Predicted objects' motion. |

## Parameters

### The `build_only` option

The `autoware_mtr` node has `build_only` option to build the TensorRT engine file from the ONNX file.

Note that although it is preferred to move all the ROS parameters in `.param.yaml` file in Autoware Universe, the `build_only` option is not moved into the `.param.yaml` file for now, because it may be used as a flag to execute the build as a pre-task.

You can execute with the following command:

```bash
ros2 launch autoware_mtr mtr.launch.xml build_only:=true
```

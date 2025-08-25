#pragma once

// C++ header
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>

// ROS header
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

//// OpenCV header
#include <opencv2/core.hpp>

// local header
#include "fcos_trt_backend/fcos_types.hpp"
#include "fcos_trt_backend/fcos_backbone.hpp"
#include "fcos_trt_backend/fcos_post_processor.hpp"


namespace fcos_object_detection
{

namespace fs = std::filesystem;

class FCOSObjectDetection : public rclcpp::Node
{
public:
  /**
   * @brief Constructor for FCOSDetection node
   */
  FCOSObjectDetection();

  /**
  * @brief Destructor for FCOSObjectDetection node
  */
  ~FCOSObjectDetection();

private:
  /**
   * @brief Initialize node parameters with validation
   * @return true if initialization successful, false otherwise
   */
  bool initialize_parameters();

    /**
   * @brief Initialize TensorRT inferencer
   * @return true if initialization successful, false otherwise
   */
  bool initialize_inferencer();

  /**
   * @brief Initialize ROS2 publishers, subscribers, and timers
   */
  void initialize_ros_components();

  /**
   * @brief Callback function for incoming images
   * @param msg Incoming image message
   */
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief Timer callback for processing images at regular intervals
   */
  void timer_callback();

  /**
   * @brief Publish object detection result in detection 2D array
   * @param detection Object detection result as Detection type
   * @param confidence_threshold Detected objects won't be published if less than this value.
   */
  void publish_detections(
    const fcos_trt_backend::Detections & detections,
    const std_msgs::msg::Header & header,
    float confidence_threshold = 0.5f);

  /**
   * @brief Publish object detection result in image
   * @param detection Object detection result as OpenCV Mat
   * @param header Original message header for timestamp consistency
   */
  void publish_detection_result_image(
    const cv::Mat & result_image,
    const std_msgs::msg::Header & header);

private:
  // ROS2 components
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr fcos_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Callback groups for parallel execution
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;
  rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

  // TensorRT inferencer
  std::shared_ptr<fcos_trt_backend::FCOSBackbone> backbone_;
  std::shared_ptr<fcos_trt_backend::FCOSPostProcessor> postprocessor_;

  // ROS2 parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string detection_topic_;
  int queue_size_;
  double processing_frequency_;
  int max_processing_queue_size_;

  fcos_trt_backend::FCOSBackbone::Config backbone_config_;
  fcos_trt_backend::FCOSPostProcessor::Config postprocessor_config_;
  fs::path engine_path_;
  std::string engine_filename_;

  // Simplified image buffer
  std::queue<sensor_msgs::msg::Image::SharedPtr> img_buff_;
  std::mutex mtx_;
  std::atomic<bool> processing_in_progress_;
};

} // namespace fcos_object_detection

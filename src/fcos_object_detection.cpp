// C++ header
#include <string>
#include <chrono>
#include <functional>
#include <exception>

// OpenCV header
#include <opencv2/highgui.hpp>

// ROS header
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>

// local header
#include "fcos_object_detection/fcos_object_detection.hpp"
#include "fcn_trt_backend/segmentation_utils.hpp"


namespace fcos_object_detection
{

FCOSObjectDetection::FCOSObjectDetection()
: Node("fcos_object_detection_node"),
  processing_in_progress_(false)
{
  // Initialize ROS2 parameters with validation
  if (!initialize_parameters()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize parameters");
    rclcpp::shutdown();
    return;
  }

  // Initialize TensorRT inferencer
  if (!initialize_inferencer()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT inferencer");
    rclcpp::shutdown();
    return;
  }

  // Initialize ROS2 components
  initialize_ros_components();

  RCLCPP_INFO(get_logger(),
    "FCN Segmentation node initialized successfully with bounded queue (max: %d)",
    max_processing_queue_size_);
}

FCOSObjectDetection::~FCOSObjectDetection()
{
  RCLCPP_INFO(get_logger(), "FCN Segmentation node shutting down");
}

bool FCOSObjectDetection::initialize_parameters()
{
  try {
    // ROS2 parameters
    input_topic_ = declare_parameter("input_topic",
      std::string("kitti/camera/color/left/image_raw"));
    output_topic_ = declare_parameter("output_topic", std::string("fcos_object_detection"));
    output_overlay_topic_ = declare_parameter("output_topic_overlay",
      std::string("fcos_object_detection_overlay"));
    queue_size_ = declare_parameter<int>("queue_size", 10);
    processing_frequency_ = declare_parameter<double>("processing_frequency", 40.0);

    // Processing queue parameter - small bounded queue for burst handling
    max_processing_queue_size_ = declare_parameter<int>("max_processing_queue_size", 3);

    // Declare and get parameters with validation
    std::string engine_package = declare_parameter("engine_package",
      std::string("fcn_trt_backend"));
    std::string engine_filename = declare_parameter("engine_filename",
      std::string("fcn_resnet101_1238x374.trt"));
    config_.width = declare_parameter<int>("width", 1238);
    config_.height = declare_parameter<int>("height", 374);
    config_.num_classes = declare_parameter<int>("num_classes", 21);
    config_.warmup_iterations = declare_parameter<int>("warmup_iterations", 2);
    config_.log_level = static_cast<fcn_trt_backend::Logger::Severity>(
      declare_parameter<int>("log_level", 3)); // Set log level

    // Validation
    if (engine_filename.empty()) {
      RCLCPP_ERROR(get_logger(), "Engine filename cannot be empty");
      return false;
    }

    if (config_.width <= 0 || config_.height <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid image dimensions: %dx%d", config_.width, config_.height);
      return false;
    }

    if (config_.num_classes <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid number of classes: %d", config_.num_classes);
      return false;
    }

    if (processing_frequency_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid processing frequency: %.2f Hz", processing_frequency_);
      return false;
    }

    if (max_processing_queue_size_ <= 0 || max_processing_queue_size_ > 10) {
      RCLCPP_ERROR(get_logger(), "Invalid max processing queue size: %d (should be 1-10)",
        max_processing_queue_size_);
      return false;
    }

    // Construct engine file path
    fs::path package_path = ament_index_cpp::get_package_share_directory(engine_package);
    engine_path_ = package_path / "engines" / engine_filename;

    RCLCPP_INFO(get_logger(), "Parameters initialized - Engine: %s, Dims: %dx%d, Classes: %d",
      engine_path_.c_str(), config_.width, config_.height, config_.num_classes);

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during parameter initialization: %s", e.what());
    return false;
  }
}

bool FCOSObjectDetection::initialize_inferencer()
{
  // Check if engine file exists
  if (!fs::exists(engine_path_)) {
    RCLCPP_ERROR(get_logger(), "Engine file does not exist: %s", engine_path_.c_str());
    return false;
  }

  try {
    segmentor = std::make_shared<fcn_trt_backend::FCNTrtBackend>(engine_path_, config_);

    if (!segmentor) {
      RCLCPP_ERROR(get_logger(), "Failed to create FCNTrtBackend instance");
      return false;
    }

    RCLCPP_INFO(get_logger(), "TensorRT inferencer initialized successfully");
    return true;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception creating FCNTrtBackend: %s", e.what());
    return false;
  }
}

void FCOSObjectDetection::initialize_ros_components()
{
  // Configure QoS profile for reliable image transport
  rclcpp::QoS image_qos(queue_size_);
  image_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  image_qos.durability(rclcpp::DurabilityPolicy::Volatile);
  image_qos.history(rclcpp::HistoryPolicy::KeepLast);

  // Create separate callback groups for parallel execution
  image_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  timer_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create subscription options with dedicated callback group
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = image_callback_group_;

  // Create subscriber with proper callback binding
  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_topic_, image_qos,
    std::bind(&FCOSObjectDetection::image_callback, this, std::placeholders::_1),
    sub_options
  );

  // Create publisher
  fcn_pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, image_qos);
  fcn_overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(output_overlay_topic_, image_qos);

  // Create timer for processing at specified frequency
  auto timer_period = std::chrono::duration<double>(1.0 / processing_frequency_);
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
    std::bind(&FCOSObjectDetection::timer_callback, this),
    timer_callback_group_
  );

  RCLCPP_INFO(get_logger(), "ROS components initialized with separate callback groups");
  RCLCPP_INFO(get_logger(), "Input: %s, Output: %s, Frequency: %.1f Hz",
    input_topic_.c_str(), output_topic_.c_str(), processing_frequency_);
}

void FCOSObjectDetection::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    // Thread-safe queue management
    std::lock_guard<std::mutex> lock(mtx_);

    // Check if queue is full
    if (img_buff_.size() >= static_cast<size_t>(max_processing_queue_size_)) {
      // Remove oldest image to make room for new one
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Processing queue full, dropping oldest image (queue size: %ld)", img_buff_.size());
      img_buff_.pop();
    }

    // Add new image to queue
    img_buff_.push(msg);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception in image callback: %s", e.what());
  }
}

void FCOSObjectDetection::timer_callback()
{
  // Skip if already processing or no subscribers
  if (processing_in_progress_.load()) {
    return;
  }

  // Check if we have subscribers before processing
  if (fcn_pub_->get_subscription_count() == 0 || fcn_overlay_pub_->get_subscription_count() == 0) {
    return;
  }

  // Get next image from queue
  sensor_msgs::msg::Image::SharedPtr msg;
  bool has_image = false;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!img_buff_.empty()) {
      msg = img_buff_.front();
      img_buff_.pop();
      has_image = true;
    }
  }

  if (!has_image) {
    return; // No image to process
  }

  // Set processing flag
  processing_in_progress_.store(true);

  try {
    // Convert ROS image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    if (!cv_ptr || cv_ptr->image.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty or invalid image");
      processing_in_progress_.store(false);
      return;
    }

    // Process the image
    cv::Mat segmentation_result = process_image(cv_ptr->image);

    if (!segmentation_result.empty()) {
      // Create overlay
      cv::Mat overlay = fcn_trt_backend::utils::create_overlay(
        cv_ptr->image, segmentation_result, 0.5f);

      // Publish results
      if (fcn_pub_->get_subscription_count() > 0) {
        publish_segmentation_result(segmentation_result, msg->header);
      }

      if (fcn_overlay_pub_->get_subscription_count() > 0) {
        publish_overlay_result(overlay, msg->header);
      }
    } else {
      RCLCPP_WARN(get_logger(), "Segmentation processing returned empty result");
    }

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during image processing: %s", e.what());
  }

  // Clear processing flag
  processing_in_progress_.store(false);
}

cv::Mat FCOSObjectDetection::process_image(const cv::Mat & input_image)
{
  if (input_image.empty()) {
    RCLCPP_WARN(get_logger(), "Input image is empty");
    return cv::Mat();
  }

  try {
    // Resize image to model input size if necessary
    cv::Mat processed_image;
    if (input_image.cols != config_.width || input_image.rows != config_.height) {
      cv::resize(input_image, processed_image, cv::Size(config_.width, config_.height), 0, 0,
        cv::INTER_LINEAR);
    } else {
      processed_image = input_image;
    }

    // Run inference
    auto segmentation = segmentor->infer(processed_image);

    if (segmentation.empty()) {
      RCLCPP_WARN(get_logger(), "Inference returned empty result");
      return cv::Mat();
    }

    // Resize back to original size if necessary
    if (segmentation.cols != input_image.cols || segmentation.rows != input_image.rows) {
      cv::Mat resized_mask;
      cv::resize(segmentation, resized_mask,
        cv::Size(input_image.cols, input_image.rows), 0, 0, cv::INTER_NEAREST);
      return resized_mask;
    }

    return segmentation;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during image processing: %s", e.what());
    return cv::Mat();
  }
}

void FCOSObjectDetection::publish_segmentation_result(
  const cv::Mat & segmentation,
  const std_msgs::msg::Header & header)
{
  try {
    // Convert OpenCV image back to ROS message
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8; // Always colored segmentation
    cv_image.image = segmentation;

    // Publish the result
    auto output_msg = cv_image.toImageMsg();
    fcn_pub_->publish(*output_msg);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during result publishing: %s", e.what());
  }
}

void FCOSObjectDetection::publish_overlay_result(
  const cv::Mat & overlay,
  const std_msgs::msg::Header & header)
{
  try {
    // Convert OpenCV image back to ROS message
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8; // Always colored segmentation
    cv_image.image = overlay;

    // Publish the result
    auto output_msg = cv_image.toImageMsg();
    fcn_overlay_pub_->publish(*output_msg);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during result publishing: %s", e.what());
  }
}

} // namespace fcos_object_detection

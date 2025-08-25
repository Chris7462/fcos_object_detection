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
#include "fcos_trt_backend/detection_utils.hpp"


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
    "FCOS object detection node initialized successfully with bounded queue (max: %d)",
    max_processing_queue_size_);
}

FCOSObjectDetection::~FCOSObjectDetection()
{
  RCLCPP_INFO(get_logger(), "FCOS object detection node shutting down");
}

bool FCOSObjectDetection::initialize_parameters()
{
  try {
    // ROS2 parameters
    input_topic_ = declare_parameter("input_topic",
      std::string("kitti/camera/color/left/image_raw"));
    output_topic_ = declare_parameter("output_topic", std::string("fcos_object_detection/image"));
    detection_topic_ = declare_parameter("detection_topic",
      std::string("fcos_object_detection/detection_array"));
    queue_size_ = declare_parameter<int>("queue_size", 10);

    processing_frequency_ = declare_parameter<double>("processing_frequency", 40.0);
    if (processing_frequency_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid processing frequency: %.2f Hz", processing_frequency_);
      return false;
    }

    // Processing queue parameter - small bounded queue for burst handling
    max_processing_queue_size_ = declare_parameter<int>("max_processing_queue_size", 3);
    if (max_processing_queue_size_ <= 0 || max_processing_queue_size_ > 10) {
      RCLCPP_ERROR(get_logger(), "Invalid max processing queue size: %d (should be 1-10)",
        max_processing_queue_size_);
      return false;
    }

    // Declare and get parameters with validation
    std::string engine_package = declare_parameter("engine_package",
      std::string("fcos_trt_backend"));
    std::string engine_filename = declare_parameter("engine_filename",
      std::string("fcos_resnet50_fpn_374x1238.engine"));
    if (engine_filename.empty()) {
      RCLCPP_ERROR(get_logger(), "Engine filename cannot be empty");
      return false;
    }

    // Construct engine file path
    fs::path package_path = ament_index_cpp::get_package_share_directory(engine_package);
    engine_path_ = package_path / "engines" / engine_filename;
    RCLCPP_INFO(get_logger(), "Parameters initialized - Loading engine form: %s.",
      engine_path_.c_str());

    // Backbone config
    backbone_config_.height = declare_parameter<int>("backbone_config.height", 374);
    backbone_config_.width = declare_parameter<int>("backbone_config.width", 1238);
    backbone_config_.warmup_iterations =
      declare_parameter<int>("backbone_config.warmup_iterations", 2);
    backbone_config_.log_level = static_cast<fcos_trt_backend::Logger::Severity>(
      declare_parameter<int>("backbone_config.log_level", 3)); // Set log level

    if (backbone_config_.width <= 0 || backbone_config_.height <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid image dimensions: %dx%d",
        backbone_config_.width, backbone_config_.height);
      return false;
    }

    // PostProcessor config
    postprocessor_config_.score_thresh =
      declare_parameter<float>("postprocessor_config.score_thresh", 0.2f);
    postprocessor_config_.nms_thresh =
      declare_parameter<float>("postprocessor_config.nms_thresh", 0.6f);
    postprocessor_config_.detections_per_img =
      declare_parameter<int>("postprocessor_config.detections_per_img", 100);
    postprocessor_config_.topk_candidates =
      declare_parameter<int>("postprocessor_config.topk_candidates", 1000);

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
    backbone_ = std::make_shared<fcos_trt_backend::FCOSBackbone>(engine_path_, backbone_config_);
    postprocessor_ = std::make_shared<fcos_trt_backend::FCOSPostProcessor>(postprocessor_config_);

    if (!backbone_) {
      RCLCPP_ERROR(get_logger(), "Failed to create FCOSBackbone instance");
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
  fcos_pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, image_qos);
  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(detection_topic_, queue_size_);

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

    // Backbone inference
    auto head_outputs = backbone_->infer(cv_ptr->image);
    auto detection_results = postprocessor_->postprocess_detections(
      head_outputs, cv_ptr->image.rows, cv_ptr->image.cols);

    if (det_pub_->get_subscription_count() > 0) {
      publish_detections(detection_results, msg->header, 0.5f);
    }

    // Plot detection results
    cv::Mat image_for_plot = fcos_trt_backend::utils::plot_detections(
      cv_ptr->image, detection_results, 0.5f);

    if (fcos_pub_->get_subscription_count() > 0) {
      publish_detection_result_image(image_for_plot, msg->header);
    }

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during image processing: %s", e.what());
  }

  // Clear processing flag
  processing_in_progress_.store(false);
}

void FCOSObjectDetection::publish_detections(
  const fcos_trt_backend::Detections & detections,
  const std_msgs::msg::Header & header,
  float confidence_threshold)
{
  // Input validation
  if (detections.boxes.empty()) {
    // Publish empty message to maintain message flow
    vision_msgs::msg::Detection2DArray empty_msg;
    empty_msg.header = header;
    det_pub_->publish(empty_msg);
    return;
  }

  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;

  size_t published_count = 0;
  for (size_t i = 0; i < detections.boxes.size(); ++i) {
    if (detections.scores[i] >= confidence_threshold) {
      // Fill bbox
      vision_msgs::msg::Detection2D det;
      const auto & box = detections.boxes[i];
      det.bbox.center.position.x = box.x + box.width / 2.0f;
      det.bbox.center.position.y = box.y + box.height / 2.0f;
      det.bbox.size_x = box.width;
      det.bbox.size_y = box.height;

      // Fill classification result
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = fcos_trt_backend::utils::get_class_name(detections.labels[i]);
      hyp.hypothesis.score = static_cast<double>(detections.scores[i]);

      det.results.emplace_back(std::move(hyp));
      msg.detections.emplace_back(std::move(det));
      ++published_count;
    }
  }

  try {
    det_pub_->publish(msg);
    RCLCPP_DEBUG(get_logger(), "Published %zu/%zu detections (confidence >= %.2f)",
      published_count, detections.boxes.size(), confidence_threshold);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to publish detections: %s", e.what());
  }
}

void FCOSObjectDetection::publish_detection_result_image(
  const cv::Mat & result_image,
  const std_msgs::msg::Header & header)
{
  try {
    // Convert OpenCV image back to ROS message
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8; // Always colored segmentation
    cv_image.image = result_image;

    // Publish the result
    auto output_msg = cv_image.toImageMsg();
    fcos_pub_->publish(*output_msg);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during result publishing: %s", e.what());
  }
}

} // namespace fcos_object_detection

// #include <so3_math.h>
#include <malloc.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "common_lib.h"
#include "li_initialization.h"

using namespace std;

#define PUBFRAME_PERIOD (20)

const float MOV_THRESHOLD = 1.5f;

string root_dir = ROOT_DIR;

int time_log_counter = 0;  //, publish_count = 0;

bool init_map = false, flg_first_scan = true;

// Time Log Variables
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool flg_reset = false, flg_exit = false;

// surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
std::deque<PointCloudXYZI::Ptr> depth_feats_world;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

auto logger = rclcpp::get_logger("laserMapping");

//按下ctrl+c后唤醒所有线程
void SigHandle(int sig)
{
  flg_exit = true;
  RCLCPP_WARN(logger, "catch sig %d", sig);
  sig_buffer.notify_all();  all();    //  会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞。
}

inline void dump_lio_state_to_log(FILE * fp)
{
  V3D rot_ang;
  if (!use_imu_as_input) {
    rot_ang = SO3ToEuler(kf_output.x_.rot);
  } else {
    rot_ang = SO3ToEuler(kf_input.x_.rot);
  }

  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle
  if (use_imu_as_input) {
    fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1),
            kf_input.x_.pos(2));                 // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // omega
    fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1),
            kf_input.x_.vel(2));                 // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // Acc
    fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1),
            kf_input.x_.bg(2));  // Bias_g
    fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1),
            kf_input.x_.ba(2));  // Bias_a
    fprintf(
      fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1),
      kf_input.x_.gravity(2));  // Bias_a
  } else {
    fprintf(
      fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1),
      kf_output.x_.pos(2));                      // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // omega
    fprintf(
      fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1),
      kf_output.x_.vel(2));                      // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // Acc
    fprintf(
      fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1),
      kf_output.x_.bg(2));  // Bias_g
    fprintf(
      fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1),
      kf_output.x_.ba(2));  // Bias_a
    fprintf(
      fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1),
      kf_output.x_.gravity(2));  // Bias_a
  }
  fprintf(fp, "\r\n");
  fflush(fp);
}

// 点云从Lidar系转到IMU系
void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu;
  if (extrinsic_est_en) {
    if (!use_imu_as_input) {  // use_imu_as_input 在 launch 文件中默认为 false
      p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
    } else {
      p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
    }
  } else {
    p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
  }
  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

//根据最新估计位姿 增量添加点云到map
void MapIncremental() //地图的增量更新，主要完成对ikd-tree的地图建立
{
  PointVector points_to_add;  //需要加入到ikd-tree中的点云
  int cur_pts = feats_down_world->size(); //加入ikd-tree时，不需要降采样的点云
  points_to_add.reserve(cur_pts); //构建的地图点

  for (size_t i = 0; i < cur_pts; ++i) {
    /* decide if need add to map */
    PointType & point_world = feats_down_world->points[i];  // 当前点云
    //判断是否有关键点需要加入到地图中
    if (!Nearest_Points[i].empty()) {
      const PointVector & points_near = Nearest_Points[i];  //获取附近的点云

      Eigen::Vector3f center =
        ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) *
        filter_size_map_min;  // 获取当前点云所属的格点中心
      bool need_add = true;   // 是否需要加入地图
      // 判断当前点的 NUM_MATCH_POINTS 个邻近点与包围盒中心的范围
      for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
        // 当前点云是否与附近的点云的距离
        Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
        // 若三个方向距离都小于地图栅格半轴长，就不添加该点
        if (
          fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
          fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
          fabs(dis_2_center.z()) < 0.5 * filter_size_map_min) {
          need_add = false;
          break;
        }
      }
      if (need_add) {
        points_to_add.emplace_back(point_world);
      }
    } 
    else {
      points_to_add.emplace_back(point_world);
    }
  }
  ivox_->AddPoints(points_to_add);
}

void publish_init_map(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudFullRes)
{
  int size_init_map = init_feats_world->size();

  sensor_msgs::msg::PointCloud2 laserCloudmsg;

  pcl::toROSMsg(*init_feats_world, laserCloudmsg);

  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
  laserCloudmsg.header.frame_id = "lidar_odom";
  pubLaserCloudFullRes->publish(laserCloudmsg);
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudFullRes)
{
  if (scan_pub_en) {
    PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
    int size = laserCloudFullRes->points.size();

    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      laserCloudWorld->points[i].x = feats_down_world->points[i].x;
      laserCloudWorld->points[i].y = feats_down_world->points[i].y;
      laserCloudWorld->points[i].z = feats_down_world->points[i].z;
      laserCloudWorld->points[i].intensity =
        feats_down_world->points[i].intensity;  // feats_down_world->points[i].y; //
    }
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "lidar_odom";
    pubLaserCloudFullRes->publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
  }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
     2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en) {
    int size = feats_down_world->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      laserCloudWorld->points[i].x = feats_down_world->points[i].x;
      laserCloudWorld->points[i].y = feats_down_world->points[i].y;
      laserCloudWorld->points[i].z = feats_down_world->points[i].z;
      laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
    }

    *pcl_wait_save += *laserCloudWorld;

    static int scan_wait_num = 0;
    scan_wait_num++;
    if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
      pcd_index++;
      string all_points_dir(
        string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      cout << "current scan saved to /PCD/" << all_points_dir << endl;
      pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
      pcl_wait_save->clear();
      scan_wait_num = 0;
    }
  }
}

// 把去畸变的雷达系下的点云转到IMU系
void publish_frame_body(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudFull_body)
{
  int size = feats_undistort->points.size();
  //创建一个点云用于存储转换到IMU系的点云
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
     //转换到IMU坐标系
    pointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);   // 将点云转换为ROS消息
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);  // 设置消息的时间戳为当前雷达结束时间戳
  laserCloudmsg.header.frame_id = "livox_frame";  //设置消息的参考坐标系为"livox_frame"。
  pubLaserCloudFull_body->publish(laserCloudmsg);
  // publish_count -= PUBFRAME_PERIOD;
}

//根据不同的输入源设置一个姿态信息（位置和方向）
//将姿态信息（位置和方向）填充到一个给定的对象 out 中
template <typename T>
void set_posestamp(T & out)
{
  if (!use_imu_as_input) {  //从kf_output获取姿态信息
    out.position.x = kf_output.x_.pos(0);
    out.position.y = kf_output.x_.pos(1);
    out.position.z = kf_output.x_.pos(2);
    /*将 kf_output 中的方向信息（四元数）赋值给 out 的 orientation 成员*/
    Eigen::Quaterniond q(kf_output.x_.rot);
    out.orientation.x = q.coeffs()[0];
    out.orientation.y = q.coeffs()[1];
    out.orientation.z = q.coeffs()[2];
    out.orientation.w = q.coeffs()[3];
  } else {  //  从kf_input获取姿态信息
    out.position.x = kf_input.x_.pos(0);
    out.position.y = kf_input.x_.pos(1);
    out.position.z = kf_input.x_.pos(2);
    Eigen::Quaterniond q(kf_input.x_.rot);
    out.orientation.x = q.coeffs()[0];
    out.orientation.y = q.coeffs()[1];
    out.orientation.z = q.coeffs()[2];
    out.orientation.w = q.coeffs()[3];
  }
}

/**
 * @brief 发布里程计数据并广播从“lidar_odom”到“base_link”的变换。
 *
 * 此函数发布里程计数据到ROS主题，并广播从“lidar_odom”坐标系到“base_link”坐标系的变换，
 * 这对于定位机器人或车辆在地图构建或导航系统中至关重要。
 *
 * @param pubOdomAftMapped 用于发布里程计数据的智能指针。
 * @param tf_br 用于广播变换的智能指针。
 * @param tf_buffer 用于查找变换的缓冲区智能指针。
 * @param logger_ 用于记录错误信息的日志记录器。
 */
void publish_odometry(
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
  std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br,
  std::unique_ptr<tf2_ros::Buffer> & tf_buffer, rclcpp::Logger logger_)
{
  odomAftMapped.header.frame_id = "lidar_odom";
  odomAftMapped.child_frame_id = "livox_frame";
  if (publish_odometry_without_downsample) {
    odomAftMapped.header.stamp = get_ros_time(time_current);
  } 
  else {
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
  }
  set_posestamp(odomAftMapped.pose.pose);

  pubOdomAftMapped->publish(odomAftMapped);

  // Publish tf from lidar_odom to base_link
  static geometry_msgs::msg::TransformStamped livox_to_base_link_transform;
  static bool transform_acquired = false;  // Check if the transform has already been acquired
  if (!transform_acquired) {
    // Get the transform from base_link to livox_frame
    try {
      livox_to_base_link_transform =
        tf_buffer->lookupTransform("livox_frame", "base_link", odomAftMapped.header.stamp);
      transform_acquired =
        true;  // Set the flag to true indicating that the transform has been acquired
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(
        logger_, "Failed to lookup transform from base_link to livox_frame: %s", ex.what());
      return;
    }
  }

  // Create a TransformStamped message for lidar_odom to base_link
  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp = odomAftMapped.header.stamp;
  transform_stamped.header.frame_id = "lidar_odom";  // Source frame
  transform_stamped.child_frame_id = "base_link";    // Target frame

  // Calculate the transform from lidar_odom to base_link by multiplying the transforms
  tf2::Transform tf_lidar_odom_to_livox_frame;
  tf2::fromMsg(odomAftMapped.pose.pose, tf_lidar_odom_to_livox_frame);
  tf2::Transform tf_livox_frame_to_base_link;
  tf2::fromMsg(livox_to_base_link_transform.transform, tf_livox_frame_to_base_link);
  tf2::Transform tf_lidar_odom_to_base_link =
    tf_lidar_odom_to_livox_frame * tf_livox_frame_to_base_link;

  // Convert the resulting transform back to geometry_msgs::TransformStamped
  transform_stamped.transform = tf2::toMsg(tf_lidar_odom_to_base_link);

  // Publish the tf
  tf_br->sendTransform(transform_stamped);
}

void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
  set_posestamp(msg_body_pose.pose);
  // msg_body_pose.header.stamp = rclcpp::Time::now();
  msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
  msg_body_pose.header.frame_id = "lidar_odom";
  static int jjj = 0;
  jjj++;
  // if (jjj % 2 == 0) // if path is too large, the rvis will crash
  {
    path.poses.emplace_back(msg_body_pose);
    pubPath->publish(path);
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto nh = std::make_shared<rclcpp::Node>("laserMapping"); //初始化ros节点，节点名为laserMapping
  // rclcpp::AsyncSpinner spinner(0);
  // spinner.start();
  readParameters(nh);   // 从参数服务器读取参数值赋给变量（包括launch文件和launch读取的yaml文件中的参数）
  cout << "lidar_type: " << lidar_type << endl;
  ivox_ = std::make_shared<IVoxType>(ivox_options_);

  path.header.stamp = get_ros_time(lidar_end_time);
  path.header.frame_id = "lidar_odom";

  /*** variables definition for counting ***/
  int frame_num = 0;
  double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
         aver_time_solve = 0, aver_time_propag = 0;

  // 将数组point_selected_surf内元素的值全部设为true，数组point_selected_surf用于选择平面点
  memset(point_selected_surf, true, sizeof(point_selected_surf));
  // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为 filter_size_surf_min
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为 filter_size_map_min
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
  //从雷达帧坐标系到IMU坐标系的转换矩阵，即雷达坐标系到IMU坐标系的旋转矩阵和偏置向量
  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

  if (extrinsic_est_en) {
    if (!use_imu_as_input) {
      kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
      kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    } else {
      kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
      kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    }
  }

  p_imu->lidar_type = p_pre->lidar_type = lidar_type;
  p_imu->imu_en = imu_en;

  //将函数地址传入kf对象中，用于接收特定于系统的模型及其差异
  // kf_input 中的f()函数就变为get_f_input函数  get_f_output df_dx_output, h_model_output
  kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, h_model_input);
  kf_output.init_dyn_share_modified_3h(
    get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
  Eigen::Matrix<double, 24, 24> P_init;  // = MD(18, 18)::Identity() * 0.1;
  reset_cov(P_init);
  kf_input.change_P(P_init);
  Eigen::Matrix<double, 30, 30> P_init_output;  // = MD(24, 24)::Identity() * 0.01;
  reset_cov_output(P_init_output);
  kf_output.change_P(P_init_output);
  Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
  Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
  /*** debug record ***/
  FILE * fp;
  string pos_log_dir = root_dir + "/Log/pos_log.txt";
  fp = fopen(pos_log_dir.c_str(), "w");
  open_file();

  /*** ROS subscribe initialization ***/
  // ROS订阅器和发布器的定义和初始化
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
  // 雷达订阅器，订阅点云的topic
  if (p_pre->lidar_type == AVIA) {
    sub_pcl_livox_ = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      lid_topic, rclcpp::SensorDataQoS(),
      [](const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) { livox_pcl_cbk(msg); });
  } else {
    sub_pcl_pc_ = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
      lid_topic, rclcpp::SensorDataQoS(),
      [](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { standard_pcl_cbk(msg); });
  }
  // IMU的订阅器sub_imu，订阅IMU的topic
  auto sub_imu =
    nh->create_subscription<sensor_msgs::msg::Imu>(imu_topic, rclcpp::SensorDataQoS(), imu_cbk);
  // 发布当前正在处理的点云数据，topic名字为 /cloud_registered
  auto pubLaserCloudFullRes =
    nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
  auto pubLaserCloudFullRes_body =
    nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 1000);
  auto pubLaserCloudEffect =
    nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 1000);
  auto pubLaserCloudMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 1000);
  auto pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 1000);
  auto pubPath = nh->create_publisher<nav_msgs::msg::Path>("/path", 1000);
  auto plane_pub = nh->create_publisher<visualization_msgs::msg::Marker>("/planner_normal", 1000);
  auto tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(nh);
  auto tf_buffer = std::make_unique<tf2_ros::Buffer>(nh->get_clock());
  auto transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  //------------------------------------------------------------------------------------------------------
  signal(SIGINT, SigHandle);
  // 设置ROS程序主循环每次运行的时间至少为0.002秒（500Hz）
  rclcpp::Rate rate(500);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(nh);
  while (rclcpp::ok()) {
    //如果有中断产生，则结束主循环
    if (flg_exit) break;
    executor.spin_some();  // 处理当前可用的回调

    if (sync_packages(Measures)) {  //把一次的IMU和LIDAR数据打包到Measures
      if (flg_reset) {  // 判断状态，并把imu的数据清空，flg_reset 默认是 false
        RCLCPP_WARN(logger, "reset when rosbag play back");
        p_imu->Reset();
        feats_undistort.reset(new PointCloudXYZI());
        if (use_imu_as_input) {   // 使用IMU初始化
          // state_in = kf_input.get_x();
          state_in = state_input();
          kf_input.change_P(P_init);
        } else {
          // state_out = kf_output.get_x();
          state_out = state_output();
          kf_output.change_P(P_init_output);
        }
        flg_first_scan = true;
        is_first_frame = true;
        flg_reset = false;
        init_map = false;

        {
          ivox_.reset(new IVoxType(ivox_options_));
        }
      }

      if (flg_first_scan) {        // 激光雷达第一次扫描
        first_lidar_time = Measures.lidar_beg_time;
        flg_first_scan = false;
        if (first_imu_time < 1) {
          // first_imu_time = get_time_sec(imu_next.header.stamp);
          first_imu_time = get_time_sec(imu_next.header.stamp);
          printf("first imu time: %f\n", first_imu_time);
        }
        time_current = 0.0;
        if (imu_en) {
          // imu_next = *(imu_deque.front());
          kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
          kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
          // kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
          // kf_output.x_.acc *= -1;

          {
            while (Measures.lidar_beg_time >
                   get_time_sec(imu_next.header.stamp))  // if it is needed for the new map?
            {
              imu_deque.pop_front();
              if (imu_deque.empty()) {
                break;
              }
              imu_last = imu_next;
              imu_next = *(imu_deque.front());  // imu_next 存的是最后一帧IMU数据
              // imu_deque.pop();
            }
          }
        } else {
          kf_input.x_.gravity << VEC_FROM_ARRAY(gravity); // _init);
          kf_output.x_.gravity << VEC_FROM_ARRAY(gravity); //_init);
          kf_output.x_.acc << VEC_FROM_ARRAY(gravity); //_init);
          kf_output.x_.acc *= -1;
          p_imu->imu_need_init_ = false;
          // p_imu->after_imu_init_ = true;
        }
        G_m_s2 = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
      }

      double t0, t1, t2, t3, t4, t5, match_start, solve_start;
      match_time = 0;
      solve_time = 0;
      propag_time = 0;
      update_time = 0;
      t0 = omp_get_wtime();
      
      /*** downsample the feature points in a scan ***/
      t1 = omp_get_wtime();   //记录时间
      //点云下采样
      p_imu->Process(Measures, feats_undistort);
      //空间下采样
      if (space_down_sample) {  //space_down_sample默认为true
        downSizeFilterSurf.setInputCloud(feats_undistort);  // 获取去畸变后的点云数据作为输入
        downSizeFilterSurf.filter(*feats_down_body);  //滤波降采样后的点云数据作为输出
        // 按时间排序
        // 将一次扫描中的点云数据按照时间从小到大排序
        sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
      } else {
        feats_down_body = Measures.lidar;
        sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
      }
      {
          time_seq = time_compressing<int>(feats_down_body);
          feats_down_size = feats_down_body->points.size();
      }
    
      if (!p_imu->after_imu_init_) {
        if (!p_imu->imu_need_init_) {
          V3D tmp_gravity;
          if (imu_en) {
            tmp_gravity = - p_imu->mean_acc / p_imu->mean_acc.norm() * G_m_s2;
          } else {
            tmp_gravity << VEC_FROM_ARRAY(gravity_init);
            p_imu->after_imu_init_ = true;
          }
          // V3D tmp_gravity << VEC_FROM_ARRAY(gravity_init);
          M3D rot_init;
          p_imu->Set_init(tmp_gravity, rot_init);
          kf_input.x_.rot = rot_init;
          kf_output.x_.rot = rot_init;
          // kf_input.x_.rot; //.normalize();
          // kf_output.x_.rot; //.normalize();
          kf_output.x_.acc = - rot_init.transpose() * kf_output.x_.gravity;
        } else {
          continue;
        }
      }
      /*** initialize the map ***/
      //构建kd树
      if (!init_map) {
        feats_down_world->resize(feats_undistort->size());
        for (int i = 0; i < feats_undistort->size(); i++) {
          {
            pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
          }
        }
        //将转换到世界坐标系的点云加入到init_feats_world 中
        for (size_t i = 0; i < feats_down_world->size(); i++) {
          init_feats_world->points.emplace_back(feats_down_world->points[i]);
        }
        //等待构建地图
        if (init_feats_world->size() < init_map_size) {
          init_map = false;
        } else {
          ivox_->AddPoints(init_feats_world->points);
          publish_init_map(pubLaserCloudMap);  //(pubLaserCloudFullRes);
          
          init_feats_world.reset(new PointCloudXYZI());
          init_map = true;
        }
        continue;
      }

      /*** ICP and Kalman filter update ***/
      /* 卡尔曼滤波和ICP更新 */
      normvec->resize(feats_down_size);
      feats_down_world->resize(feats_down_size);

      Nearest_Points.resize(feats_down_size); // 存储近邻点的vector，将降采样处理后的点云用于搜索最近点

      t2 = omp_get_wtime();   // 初始化t2为当前时间

      /*** iterated state estimation ***/
      crossmat_list.reserve(feats_down_size);
      pbody_list.reserve(feats_down_size);
      // pbody_ext_list.reserve(feats_down_size);

      // 对于扫描中的每个点，将点云转换到世界坐标系下，并记录转换的旋转矩阵，用于计算雅克比矩阵
      for (size_t i = 0; i < feats_down_body->size(); i++) {
        V3D point_this(
          feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
        pbody_list[i] = point_this;
        if (!extrinsic_est_en)   // 是否估计外参，avia.yaml中是false
                                 // 对于激进的运动，将此变量设置为 false
        // {
        //     if (!use_imu_as_input)
        //     {
        //         point_this = kf_output.x_.offset_R_L_I * point_this +
        //         kf_output.x_.offset_T_L_I;
        //     }
        //     else
        //     {
        //         point_this = kf_input.x_.offset_R_L_I * point_this +
        //         kf_input.x_.offset_T_L_I;
        //     }
        // }
        // else
        {
          point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
          M3D point_crossmat;
          point_crossmat << SKEW_SYM_MATRX(point_this);
          crossmat_list[i] = point_crossmat;
        }
      }
      if (!use_imu_as_input) {  // use_imu_as_input 默认为true
        bool imu_upda_cov = false;   // 是否需要更新imu的协方差
        effct_feat_num = 0;
        /**** point by point update ****/
        if (time_seq.size() > 0) {
          double pcl_beg_time = Measures.lidar_beg_time;
          idx = -1;
          for (k = 0; k < time_seq.size(); k++) {
            PointType & point_body = feats_down_body->points[idx + time_seq[k]];

            time_current = point_body.curvature / 1000.0 + pcl_beg_time;

            if (is_first_frame) {
              if (imu_en) {
                while (time_current > get_time_sec(imu_next.header.stamp)) {
                  imu_deque.pop_front();
                  if (imu_deque.empty()) break;
                  imu_last = imu_next;
                  imu_next = *(imu_deque.front());
                }
                angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y,
                  imu_last.angular_velocity.z;
                acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y,
                  imu_last.linear_acceleration.z;
              }
              is_first_frame = false;
              imu_upda_cov = true;
              time_update_last = time_current;
              time_predict_last_const = time_current;
            }
            if (imu_en && !imu_deque.empty()) {
              bool last_imu = get_time_sec(imu_next.header.stamp) ==
                              get_time_sec(imu_deque.front()->header.stamp);
              while (get_time_sec(imu_next.header.stamp) < time_predict_last_const &&
                     !imu_deque.empty()) {
                if (!last_imu) {
                  imu_last = imu_next;
                  imu_next = *(imu_deque.front());
                  break;
                } else {
                  imu_deque.pop_front();
                  if (imu_deque.empty()) break;
                  imu_last = imu_next;
                  imu_next = *(imu_deque.front());
                }
              }
              bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
              while (imu_comes) {
                imu_upda_cov = true;
                angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y,
                  imu_next.angular_velocity.z;
                acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y,
                  imu_next.linear_acceleration.z;

                /*** covariance update ***/
                double dt = get_time_sec(imu_next.header.stamp) - time_predict_last_const;
                kf_output.predict(dt, Q_output, input_in, true, false);
                time_predict_last_const = get_time_sec(imu_next.header.stamp);  // big problem

                {
                  double dt_cov = get_time_sec(imu_next.header.stamp) - time_update_last;

                  if (dt_cov > 0.0) {
                    time_update_last = get_time_sec(imu_next.header.stamp);
                    double propag_imu_start = omp_get_wtime();

                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                    propag_time += omp_get_wtime() - propag_imu_start;
                    double solve_imu_start = omp_get_wtime();
                    kf_output.update_iterated_dyn_share_IMU();
                    solve_time += omp_get_wtime() - solve_imu_start;
                  }
                }
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());
                imu_comes = time_current > get_time_sec(imu_next.header.stamp);
              }
            }
            if (flg_reset) {
              break;
            }

            double dt = time_current - time_predict_last_const;
            double propag_state_start = omp_get_wtime();
            if (!prop_at_freq_of_imu) {
              double dt_cov = time_current - time_update_last;
              if (dt_cov > 0.0) {
                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                time_update_last = time_current;
              }
            }
            kf_output.predict(dt, Q_output, input_in, true, false);
            propag_time += omp_get_wtime() - propag_state_start;
            time_predict_last_const = time_current;
            double t_update_start = omp_get_wtime();

            if (feats_down_size < 1) {
              RCLCPP_WARN(logger, "No point, skip this scan!\n");
              idx += time_seq[k];
              continue;
            }
            if (!kf_output.update_iterated_dyn_share_modified()) {
              idx = idx + time_seq[k];
              continue;
            }
            solve_start = omp_get_wtime();

            if (publish_odometry_without_downsample) {
              /******* Publish odometry *******/

              publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, nh->get_logger());
              if (runtime_pos_log) {
                euler_cur = SO3ToEuler(kf_output.x_.rot);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                         << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " "
                         << kf_output.x_.vel.transpose() << " " << kf_output.x_.omg.transpose()
                         << " " << kf_output.x_.acc.transpose() << " "
                         << kf_output.x_.gravity.transpose() << " " << kf_output.x_.bg.transpose()
                         << " " << kf_output.x_.ba.transpose() << " "
                         << feats_undistort->points.size() << endl;
              }
            }

            for (int j = 0; j < time_seq[k]; j++) {
              PointType & point_body_j = feats_down_body->points[idx + j + 1];
              PointType & point_world_j = feats_down_world->points[idx + j + 1];
              pointBodyToWorld(&point_body_j, &point_world_j);
            }

            solve_time += omp_get_wtime() - solve_start;

            update_time += omp_get_wtime() - t_update_start;
            idx += time_seq[k];
            // cout << "pbp output effect feat num:" << effct_feat_num << endl;
          }
        } else {
          if (!imu_deque.empty()) {
            imu_last = imu_next;
            imu_next = *(imu_deque.front());

            while (get_time_sec(imu_next.header.stamp) > time_current &&
                   ((get_time_sec(imu_next.header.stamp) <
                     Measures.lidar_beg_time + lidar_time_inte))) {  // >= ?
              if (is_first_frame) { //判断是否第一帧
                {
                  {
                    while (get_time_sec(imu_next.header.stamp) <
                           Measures.lidar_beg_time + lidar_time_inte) {
                      // meas.imu.emplace_back(imu_deque.front()); should add to
                      // initialization
                      imu_deque.pop_front();
                      if (imu_deque.empty()) break;
                      imu_last = imu_next;
                      imu_next = *(imu_deque.front());
                    }
                  }
                  break;
                }
                angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y,
                  imu_last.angular_velocity.z;

                acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y,
                  imu_last.linear_acceleration.z;

                imu_upda_cov = true;
                time_update_last = time_current;
                time_predict_last_const = time_current;

                is_first_frame = false;
              }
              time_current = get_time_sec(imu_next.header.stamp);

              if (!is_first_frame) {  //判断是否第一帧
                double dt = time_current - time_predict_last_const;
                {
                  double dt_cov = time_current - time_update_last;
                  if (dt_cov > 0.0) {
                    kf_output.predict(dt_cov, Q_output, input_in, false, true);
                    time_update_last = time_current;
                  }
                  kf_output.predict(dt, Q_output, input_in, true, false);
                }

                time_predict_last_const = time_current;

                angvel_avr << imu_next.angular_velocity.x,
                    imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                acc_avr << imu_next.linear_acceleration.x,
                    imu_next.linear_acceleration.y,
                    imu_next.linear_acceleration.z;
                // acc_avr_norm = acc_avr * G_m_s2 / acc_norm;
                kf_output.update_iterated_dyn_share_IMU();
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());
              } else {
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());
              }
            }
          }
        }
      } else {  //直接到这里执行
        bool imu_prop_cov = false;  //是否需要更新imu的协方差
        effct_feat_num = 0;


        if (time_seq.size() > 0) {
          // 首先设置pcl_beg_time为Measures.lidar_beg_time，idx为-1
          double pcl_beg_time = Measures.lidar_beg_time;
          idx = -1;
          for (k = 0; k < time_seq.size(); k++) {
            PointType & point_body = feats_down_body->points[idx + time_seq[k]];
            // 找到对应的点并计算出当前时间time_current
            // 计算当前雷达点的时刻，毫秒除1000变为纳秒，雷达开始时刻的时间戳单位为纳秒ns
            time_current = point_body.curvature / 1000.0 + pcl_beg_time;
            if (is_first_frame) {  // 如果是第一帧雷达扫描
              while (time_current > get_time_sec(imu_next.header.stamp)) {
                // 将IMU数据出队，直到雷达开始时刻在IMU时间戳之前
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());  //imu_next为队列中最前面的一帧IMU数据，也就是取出来的最后一帧IMU数据
              }
              imu_prop_cov = true;

              is_first_frame = false;
              t_last = time_current;  // 记录第一个雷达点的时间
              time_update_last = time_current;  // 记录第一个雷达点的时间
              {
                input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y,
                  imu_last.angular_velocity.z;
                input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y,
                  imu_last.linear_acceleration.z;
                input_in.acc = input_in.acc * G_m_s2 / acc_norm;
              }
            }

            while (time_current > get_time_sec(imu_next.header.stamp))  // && !imu_deque.empty())
            {
              imu_deque.pop_front();

              input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y,
                imu_last.angular_velocity.z;
              input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y,
                imu_last.linear_acceleration.z;
              input_in.acc = input_in.acc * G_m_s2 / acc_norm;
              double dt = get_time_sec(imu_last.header.stamp) - t_last;

              double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
              if (dt_cov > 0.0) {
                kf_input.predict(dt_cov, Q_input, input_in, false, true);
                time_update_last = get_time_sec(imu_last.header.stamp);  // time_current;
              }
              kf_input.predict(dt, Q_input, input_in, true, false);
              t_last = get_time_sec(imu_last.header.stamp);
              imu_prop_cov = true;

              if (imu_deque.empty()) break;
              imu_last = imu_next;
              imu_next = *(imu_deque.front());
              // imu_upda_cov = true;
            }
            if (flg_reset) {
              break;
            }
            double dt = time_current - t_last;
            t_last = time_current;
            double propag_start = omp_get_wtime();

            if (!prop_at_freq_of_imu) {
              double dt_cov = time_current - time_update_last;
              if (dt_cov > 0.0) {
                kf_input.predict(dt_cov, Q_input, input_in, false, true);
                time_update_last = time_current;
              }
            }
            kf_input.predict(dt, Q_input, input_in, true, false);

            propag_time += omp_get_wtime() - propag_start;

            double t_update_start = omp_get_wtime();

            if (feats_down_size < 1) {
              RCLCPP_WARN(logger, "No point, skip this scan!\n");

              idx += time_seq[k];
              continue;
            }
            if (!kf_input.update_iterated_dyn_share_modified()) {
              idx = idx + time_seq[k];
              continue;
            }

            solve_start = omp_get_wtime();

            if (publish_odometry_without_downsample) {
              /******* Publish odometry *******/

              publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, nh->get_logger());
              if (runtime_pos_log) {
                euler_cur = SO3ToEuler(kf_input.x_.rot);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                         << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " "
                         << kf_input.x_.vel.transpose() << " " << kf_input.x_.bg.transpose() << " "
                         << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose()
                         << " " << feats_undistort->points.size() << endl;
              }
            }

            for (int j = 0; j < time_seq[k]; j++) {
              PointType & point_body_j = feats_down_body->points[idx + j + 1];
              PointType & point_world_j = feats_down_world->points[idx + j + 1];
              pointBodyToWorld(&point_body_j, &point_world_j);
            }
            solve_time += omp_get_wtime() - solve_start;

            update_time += omp_get_wtime() - t_update_start;
            idx = idx + time_seq[k];
          }
        } else {
          if (!imu_deque.empty()) {
            imu_last = imu_next;
            imu_next = *(imu_deque.front());
            while (get_time_sec(imu_next.header.stamp) > time_current &&
                   ((get_time_sec(imu_next.header.stamp) <
                     Measures.lidar_beg_time + lidar_time_inte))) {  // >= ?
              if (is_first_frame) {
                {
                  {
                    while (get_time_sec(imu_next.header.stamp) <
                           Measures.lidar_beg_time + lidar_time_inte) {
                      imu_deque.pop_front();
                      if (imu_deque.empty()) break;
                      imu_last = imu_next;
                      imu_next = *(imu_deque.front());
                    }
                  }

                  break;
                }
                imu_prop_cov = true;

                t_last = time_current;
                time_update_last = time_current;
                input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y,
                  imu_last.angular_velocity.z;
                input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y,
                  imu_last.linear_acceleration.z;
                input_in.acc = input_in.acc * G_m_s2 / acc_norm;

                is_first_frame = false;
              }
              time_current = get_time_sec(imu_next.header.stamp);

              if (!is_first_frame) {
                double dt = time_current - t_last;

                double dt_cov = time_current - time_update_last;
                if (dt_cov > 0.0) {
                  // kf_input.predict(dt_cov, Q_input, input_in, false, true);
                  time_update_last = get_time_sec(imu_next.header.stamp);  // time_current;
                }
                // kf_input.predict(dt, Q_input, input_in, true, false);

                t_last = get_time_sec(imu_next.header.stamp);

                input_in.gyro << imu_next.angular_velocity.x, imu_next.angular_velocity.y,
                  imu_next.angular_velocity.z;
                input_in.acc << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y,
                  imu_next.linear_acceleration.z;
                input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());
              } else {
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_last = imu_next;
                imu_next = *(imu_deque.front());
              }
            }
          }
        }
      }
      // M3D rot_cur_lidar;
      // {
      //     rot_cur_lidar = state.rot_end;
      // }
      // euler_cur = RotMtoEuler(rot_cur_lidar);
      // geoQuat = tf::createQuaternionMsgFromRollPitchYaw
      //                     (euler_cur(0), euler_cur(1), euler_cur(2));
      /******* Publish odometry downsample *******/
      // 开关高频里程计，配置文件中设为 true 则打开高频里程计
      // 开启高频里程计后不执行这里
      if (!publish_odometry_without_downsample) {
        publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, nh->get_logger());
      }

      /*** add the feature points to map ***/
      t3 = omp_get_wtime();

      if (feats_down_size > 4) {
        /* 向映射ikdtree添加特征点 */
        MapIncremental();
      }

      t5 = omp_get_wtime();
      /******* Publish points *******/
      /* 发布轨迹和点 */
      if (path_en) publish_path(pubPath);
      // 发布当前正在扫描的点云的topic 或者 保存点云为PCD文件
      if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFullRes);
      // 发布经过运动畸变校正注册到IMU坐标系的点云的topic
      if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);

      /*** Debug variables Logging ***/
      if (runtime_pos_log) {
        frame_num++;
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        {
          aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num;
        }
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
        aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
        T1[time_log_counter] = Measures.lidar_beg_time;
        s_plot[time_log_counter] = t5 - t0;
        s_plot2[time_log_counter] = feats_undistort->points.size();
        s_plot3[time_log_counter] = aver_time_consu;
        time_log_counter++;
        printf(
          "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave "
          "match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: "
          "%0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",
          t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu,
          aver_time_icp, aver_time_propag);
        if (!publish_odometry_without_downsample) {
          if (!use_imu_as_input) {
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                     << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " "
                     << kf_output.x_.vel.transpose() << " " << kf_output.x_.omg.transpose() << " "
                     << kf_output.x_.acc.transpose() << " " << kf_output.x_.gravity.transpose()
                     << " " << kf_output.x_.bg.transpose() << " " << kf_output.x_.ba.transpose()
                     << " " << feats_undistort->points.size() << endl;
          } else {
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                     << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " "
                     << kf_input.x_.vel.transpose() << " " << kf_input.x_.bg.transpose() << " "
                     << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose() << " "
                     << feats_undistort->points.size() << endl;
          }
        }
        dump_lio_state_to_log(fp);
      }
    }
    rate.sleep();
  }
  //--------------------------save map-----------------------------------
  /* 1. make sure you have enough memories
     2. noted that pcd save will influence the real-time performences **/
  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    string file_name = string("scans.pcd");
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
  }
  fout_out.close();
  fout_imu_pbp.close();
  return 0;
}

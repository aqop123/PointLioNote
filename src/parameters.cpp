#include "parameters.h"

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0, time_con = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;
IVoxType::Options ivox_options_;
int ivox_nearby_type = 6;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
state_input state_in;
state_output state_out;
std::string lid_topic, imu_topic;
bool prop_at_freq_of_imu = true, check_satu = true, con_frame = false, cut_frame = false;
bool use_imu_as_input = false, space_down_sample = true, publish_odometry_without_downsample = false;
int  init_map_size = 10, con_frame_num = 1;
double match_s = 81, satu_acc, satu_gyro, cut_frame_time_interval = 0.1;
float  plane_thr = 0.1f;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5, fov_deg = 180;
// double cube_len = 2000; 
float  DET_RANGE = 450;
bool   imu_en = true;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01, acc_norm;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov; 
int    lidar_type, pcd_save_interval;
std::vector<double> gravity_init, gravity;
bool   runtime_pos_log, pcd_save_en, path_en, extrinsic_est_en = true;
bool   scan_pub_en, scan_body_pub_en;
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
double time_diff_lidar_to_imu = 0.0;

double lidar_time_inte = 0.1, first_imu_time = 0.0;
int cut_frame_num = 1, orig_odom_freq = 10;
double online_refine_time = 20.0; //unit: s
bool cut_frame_init = true;

MeasureGroup Measures;

ofstream fout_out, fout_imu_pbp, fout_rtk;

//读取参数服务器的参数
void readParameters(shared_ptr<rclcpp::Node> &nh)
{
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  nh->declare_parameter<bool>("prop_at_freq_of_imu", true);
  nh->declare_parameter<bool>("use_imu_as_input", true);
  nh->declare_parameter<bool>("check_satu", true);
  nh->declare_parameter<int>("init_map_size", 100);
  nh->declare_parameter<bool>("space_down_sample", true);
  nh->declare_parameter<double>("mapping.satu_acc", 3.0);
  nh->declare_parameter<double>("mapping.satu_gyro", 35.0);
  nh->declare_parameter<double>("mapping.acc_norm", 1.0);
  nh->declare_parameter<float>("mapping.plane_thr", 0.05f);
  nh->declare_parameter<int>("point_filter_num", 2);    // 采样间隔，即每隔point_filter_num个点取1个点
  nh->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");   // 雷达点云topic名称
  nh->declare_parameter<std::string>("common.imu_topic", "/livox/imu");     // IMU的topic名称
  nh->declare_parameter<bool>("common.con_frame", false);
  nh->declare_parameter<int>("common.con_frame_num", 1);
  nh->declare_parameter<bool>("common.cut_frame", false);
  nh->declare_parameter<double>("common.cut_frame_time_interval", 0.1);
  nh->declare_parameter<double>("common.time_diff_lidar_to_imu", 0.0);
  nh->declare_parameter<double>("filter_size_surf", 0.5);
  nh->declare_parameter<double>("filter_size_map", 0.5);
  // nh.param<double>("cube_side_length",cube_len,2000);
  nh->declare_parameter<float>("mapping.det_range", 300.f);
  nh->declare_parameter<double>("mapping.fov_degree", 180);
  nh->declare_parameter<bool>("mapping.imu_en", true);
  nh->declare_parameter<bool>("mapping.extrinsic_est_en", true);
  nh->declare_parameter<double>("mapping.imu_time_inte", 0.005);
  nh->declare_parameter<double>("mapping.lidar_meas_cov", 0.1);
  nh->declare_parameter<double>("mapping.acc_cov_input", 0.1);
  nh->declare_parameter<double>("mapping.vel_cov", 20);
  nh->declare_parameter<double>("mapping.gyr_cov_input", 0.1);
  nh->declare_parameter<double>("mapping.gyr_cov_output", 0.1);
  nh->declare_parameter<double>("mapping.acc_cov_output", 0.1);
  nh->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  nh->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  nh->declare_parameter<double>("mapping.imu_meas_acc_cov", 0.1);
  nh->declare_parameter<double>("mapping.imu_meas_omg_cov", 0.1);
  nh->declare_parameter<double>("preprocess.blind", 1.0);
  nh->declare_parameter<int>("preprocess.lidar_type", 1);    // 激光雷达的类型
  nh->declare_parameter<int>("preprocess.scan_line", 16);    // 激光雷达扫描的线数（livox avia为6线）
  nh->declare_parameter<int>("preprocess.scan_rate", 10);
  nh->declare_parameter<int>("preprocess.timestamp_unit", 1);
  nh->declare_parameter<double>("mapping.match_s", 81);
  nh->declare_parameter<std::vector<double>>("mapping.gravity", {0, 0, -9.810});
  nh->declare_parameter<std::vector<double>>("mapping.gravity_init", {0, 0, -9.810});
  nh->declare_parameter<std::vector<double>>("mapping.extrinsic_T", {0, 0, 0});   // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
  nh->declare_parameter<std::vector<double>>("mapping.extrinsic_R", {1, 0, 0, 0, 1, 0, 0, 0, 1});   // 雷达相对于IMU的外参R
  nh->declare_parameter<bool>("odometry.publish_odometry_without_downsample", false);
  nh->declare_parameter<bool>("publish.path_en", true);
  nh->declare_parameter<bool>("publish.scan_publish_en", true);
  nh->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  nh->declare_parameter<bool>("runtime_pos_log_enable", false);
  nh->declare_parameter<bool>("pcd_save.pcd_save_en", false);
  nh->declare_parameter<int>("pcd_save.interval", -1);
  nh->declare_parameter<double>("mapping.lidar_time_inte", 0.1);
  nh->declare_parameter<float>("mapping.ivox_grid_resolution", 0.2);
  nh->declare_parameter<int>("ivox_nearby_type", 18);

  nh->get_parameter("prop_at_freq_of_imu", prop_at_freq_of_imu);
  nh->get_parameter("use_imu_as_input", use_imu_as_input);
  nh->get_parameter("check_satu", check_satu);
  nh->get_parameter("init_map_size", init_map_size);
  nh->get_parameter("space_down_sample", space_down_sample);
  nh->get_parameter("mapping.satu_acc", satu_acc);
  nh->get_parameter("mapping.satu_gyro", satu_gyro);
  nh->get_parameter("mapping.acc_norm", acc_norm);
  nh->get_parameter("mapping.plane_thr", plane_thr);
  nh->get_parameter("point_filter_num", p_pre->point_filter_num);
  nh->get_parameter("common.lid_topic", lid_topic);
  nh->get_parameter("common.imu_topic", imu_topic);
  nh->get_parameter("common.con_frame", con_frame);
  nh->get_parameter("common.con_frame_num", con_frame_num);
  nh->get_parameter("common.cut_frame", cut_frame);
  nh->get_parameter("common.cut_frame_time_interval", cut_frame_time_interval);
  nh->get_parameter("common.time_diff_lidar_to_imu", time_diff_lidar_to_imu);
  nh->get_parameter("filter_size_surf", filter_size_surf_min);
  nh->get_parameter("filter_size_map", filter_size_map_min);
  nh->get_parameter("mapping.det_range", DET_RANGE);
  nh->get_parameter("mapping.fov_degree", fov_deg);
  nh->get_parameter("mapping.imu_en", imu_en);
  nh->get_parameter("mapping.extrinsic_est_en", extrinsic_est_en);
  nh->get_parameter("mapping.imu_time_inte", imu_time_inte);
  nh->get_parameter("mapping.lidar_meas_cov", laser_point_cov);
  nh->get_parameter("mapping.acc_cov_input", acc_cov_input);
  nh->get_parameter("mapping.vel_cov", vel_cov);
  nh->get_parameter("mapping.gyr_cov_input", gyr_cov_input);
  nh->get_parameter("mapping.gyr_cov_output", gyr_cov_output);
  nh->get_parameter("mapping.acc_cov_output", acc_cov_output);
  nh->get_parameter("mapping.b_gyr_cov", b_gyr_cov);
  nh->get_parameter("mapping.b_acc_cov", b_acc_cov);
  nh->get_parameter("mapping.imu_meas_acc_cov", imu_meas_acc_cov);
  nh->get_parameter("mapping.imu_meas_omg_cov", imu_meas_omg_cov);
  nh->get_parameter("preprocess.blind", p_pre->blind);
  nh->get_parameter("preprocess.lidar_type", lidar_type);
  nh->get_parameter("preprocess.scan_line", p_pre->N_SCANS);
  nh->get_parameter("preprocess.scan_rate", p_pre->SCAN_RATE);
  nh->get_parameter("preprocess.timestamp_unit", p_pre->time_unit);
  nh->get_parameter("mapping.match_s", match_s);
  nh->get_parameter("mapping.gravity", gravity);
  nh->get_parameter("mapping.gravity_init", gravity_init);
  nh->get_parameter("mapping.extrinsic_T", extrinT);
  nh->get_parameter("mapping.extrinsic_R", extrinR);
  nh->get_parameter("odometry.publish_odometry_without_downsample", publish_odometry_without_downsample);
  nh->get_parameter("publish.path_en", path_en);
  nh->get_parameter("publish.scan_publish_en", scan_pub_en);
  nh->get_parameter("publish.scan_bodyframe_pub_en", scan_body_pub_en);
  nh->get_parameter("runtime_pos_log_enable", runtime_pos_log);
  nh->get_parameter("pcd_save.pcd_save_en", pcd_save_en);
  nh->get_parameter("pcd_save.interval", pcd_save_interval);
  nh->get_parameter("mapping.lidar_time_inte", lidar_time_inte);
  nh->get_parameter("mapping.lidar_meas_cov", laser_point_cov);
  nh->get_parameter("mapping.ivox_grid_resolution", ivox_options_.resolution_);
  nh->get_parameter("ivox_nearby_type", ivox_nearby_type);

  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    // LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
}

/**
 * @brief 将SO(3)旋转转换为欧拉角
 * 
 * 该函数接受一个SO(3)旋转矩阵并将其转换为对应的欧拉角表示。
 * 欧拉角表示通过三个旋转角度(x, y, z)来描述一个三维旋转。
 * 如果旋转矩阵在x轴或y轴上的投影非常小（在这种情况下认为是奇异点），
 * 则采用不同的计算方式来避免数值不稳定。
 * 
 * @param rot 一个3x3的SO(3)旋转矩阵
 * @return Eigen::Matrix<double, 3, 1> 包含三个欧拉角（以弧度为单位）的向量
 */
Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 &rot) 
{
    // 计算旋转矩阵第一列的欧几里得范数
    double sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    // 判断是否处于奇异状态，即当第一列接近零向量时
    bool singular = sy < 1e-6;
    double x, y, z;
    // 当不是奇异状态时，正常计算欧拉角
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    // 当处于奇异状态时，使用另一种方式计算欧拉角以避免数值问题
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    // 将计算得到的欧拉角组装成一个3维向量并返回
    Eigen::Matrix<double, 3, 1> ang(x, y, z);
    return ang;
}

void open_file()
{

    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"),ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

}

/**
 * 重置协方差矩阵
 * 
 * 对于一个卡尔曼滤波器或者类似处理不确定性的算法来说，协方差矩阵是一个核心组件，
 * 它描述了估计误差的不确定性。此函数的目的是重置协方差矩阵，以便在算法的某些阶段
 * 重新初始化误差估计。函数首先用一个相对较小的标量乘以单位矩阵来初始化整个矩阵，
 * 表示一个相对较小的初始误差估计。然后，函数在矩阵的特定子块上设置甚至更小的协方差值，
 * 这可能对应于算法中某些变量或状态的更高置信度。
 * 
 * @param P_init 24x24的矩阵，表示协方差矩阵被重置后将用于初始化某些算法（如卡尔曼滤波器）的状态。
 */
void reset_cov(Eigen::Matrix<double, 24, 24> & P_init)
{
    // 初始化整个协方差矩阵为一个相对较小的值
    P_init = MD(24, 24)::Identity() * 0.1;
    
    // 在矩阵的右下角3x3子块中，设置一个更小的协方差值
    P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    
    // 在矩阵的另一个子块6x6中，设置一个相对较小的协方差值
    P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;
}

void reset_cov_output(Eigen::Matrix<double, 30, 30> & P_init_output)
{
    P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    // P_init_output.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6,6)::Identity() * 0.001;
}
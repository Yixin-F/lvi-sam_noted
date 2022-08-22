# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable  # scan2map初值可用(9轴Imu才行)，当前scan中的每一时刻imu相对于起始时刻的旋转增量可用(6轴imu即可)
int64 odomAvailable

# Attitude for lidar odometry initialization
# 来自imu预积分
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Odometry 
# 来自VIS里程计
float32 odomX
float32 odomY
float32 odomZ
float32 odomRoll
float32 odomPitch
float32 odomYaw

# Odometry reset ID
int64 odomResetId

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
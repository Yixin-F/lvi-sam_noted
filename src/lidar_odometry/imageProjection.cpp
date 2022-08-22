#include "utility.h"
#include "lvi_sam/cloud_info.h"

//  > Velodyne 数据类型 （点云坐标、密度、线圈数、时间戳）
struct PointXYZIRT
{
    PCL_ADD_POINT4D ;  // > 分别有float类型的 x、y、z 还有一个对齐变量
    PCL_ADD_INTENSITY;  // > float类型的密度 
    uint16_t ring;   // > 总的线圈数, velodyne独有ring信息
    float time;   // > 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // > 保证在内存中是对齐的状态
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,
(float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
(uint16_t, ring, ring) (float, time, time)
)

// Ouster 数据类型 （点云坐标、密度、时间戳、反射率、线圈数、错误率、范围）
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     uint32_t t;
//     uint16_t reflectivity;
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

const int queueLength = 500;

class ImageProjection : public ParamServer
{
private:

    // ! 以下这两个锁是互斥关系
    std::mutex imuLock;   //  > imu锁
    std::mutex odoLock;   // > 里程计锁

    ros::Subscriber subLaserCloud; //雷达点云订阅
    ros::Publisher  pubLaserCloud; //雷达点云发布

    ros::Publisher pubExtractedCloud; //发布提取后的雷达点云
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu; //imu订阅
    std::deque<sensor_msgs::Imu> imuQueue; //imu队列

    ros::Subscriber subOdom; //里程计订阅
    std::deque<nav_msgs::Odometry> odomQueue; //里程计队列

    std::deque<sensor_msgs::PointCloud2> cloudQueue; //点云队列
    sensor_msgs::PointCloud2 currentCloudMsg; //当前的点云信息

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;//点云对应imu的索引
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse; //初始的位姿

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat; // > range image

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lvi_sam::cloud_info cloudInfo; // > 自定义消息类型
    double timeScanCur;
    double timeScanNext;
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
            deskewFlag(0)
    {
        // 重新订阅了imu消息，与imuPreintegration.cpp无关
        // > 用于点云旋转去畸变和为scan2map提供初值的imu数据 -> 用在在VIS失效
        subImu        = nh.subscribe<sensor_msgs::Imu>        (imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());

        // ! 激光惯性子系统(LIS)与视觉惯性子系统的关系 -> 紧耦合关系
        // > 1) VIS利用LIS的初始化估计来初始化，包括bias、位姿等，快且鲁棒
        // > 2) VIS的精度提高依赖于LIS深度信息的提取
        // ? 3) LIS利用VIS结果来scan2map初始化，如果VIS失败，则直接使用imu的值
        // > 4) 回环检测先经VIS，再LIS优化匹配
        // > 5) VIS和LIS都可以在一方失效后独立运行(VIS跟踪失败或LIS特征缺失)，提高整个系统的鲁棒性

        // ? 订阅vins的imu预积分里程计？
        // ! 原lio-sam此处为订阅imu增量式里程计：来自IMUPreintegration发布的增量式里程计话题(前一帧激光帧优化基础上), 这个话题已经不存在了，它用于去畸变和为scan2map提供初值
        // ! subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // > 用于点云平移去畸变和scan2map提供初值的视觉里程计数据  -> VIS不失效时首选(LIS初始化时仍使用imu来提供scan2map的初值，因为LIS先初始化)
        subOdom       = nh.subscribe<nav_msgs::Odometry>      (PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>      (PROJECT_NAME + "/lidar/deskew/cloud_info", 5);

        allocateMemory();
        resetParameters();

        // > setVerbosityLevel用于设置控制台输出的信息，这里是指汇报任何pcl错误信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);//将所有点云数据安装行列从小到大的顺序存储在fullCloud内

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    //对每个获取的lidar message进行参数重置
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();

        // reset range matrix for range image projection
        // 雷达深度图 Range Image
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    //析构函数
    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);  // 将imu的三个轴的线加速度和角加速度的信息旋转到以lidar为中心的坐标系

        // > lock_guard对象不以任何方式管理互斥对象的生存期，不会因为对象抛出异常而导致死锁，简单来讲就是自动管理上锁时间，优于.lock()和.unlock()的组合
        // > 但是对于一个锁，仍需要两个代码地点的对应关系，deskewInfo()中体现到了，读取与去畸变同时使用到了imuQueue中的内存，为了防止错误，需要用互斥锁进行互斥
        std::lock_guard<std::mutex> lock1(imuLock);

        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    // > 点云操作主函数
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // Step 1 检查队列里面的点云数量是否满足要求 并做一些前置操作
        if (!cachePointCloud(laserCloudMsg))
            return;

        // Step 2 对IMU和视觉里程计去畸变
        if (!deskewInfo())
            return;

        // Step 3 获取雷达深度图
        projectPointCloud();

        // Step 4 点云提取
        cloudExtraction();

        // Step 5 发布点云
        publishClouds();

        // Step 6 重置参数
        resetParameters();
    }

    //队列里面的点云数量是否满足要求
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);

        if (cloudQueue.size() <= 2)//队列里面的点云数量小于3
            return false;
        else
        {
            currentCloudMsg = cloudQueue.front();
            cloudQueue.pop_front();//将队列头的点云弹出

            cloudHeader = currentCloudMsg.header;
            timeScanCur = cloudHeader.stamp.toSec();//当前点云扫描的起始时间戳
            timeScanNext = cloudQueue.front().header.stamp.toSec();//下一帧点云的时间戳
        }

        // convert cloud
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn); //读取ros信息转为pcl信息

        // check dense flag
        if (laserCloudIn->is_dense == false)///表示点云里面没有去除无效点（NaN）
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // > 检查 点云是否包含ring通道, useful
        // 该部分主要用来计算rowIdn
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        //yaml文件中 timeField: "time"    # point timestamp field, Velodyne - "time", Ouster - "t"
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {

                //表示当前具有时间戳信息
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    // > 点云去畸变主函数
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");  // imuQueue.front().header.stamp.toSec() 这种情况下是等不来的，虽然不太可能发生这种情况
            return false;
        }

        // >  IMU旋转去畸变
        imuDeskewInfo();

        //  > VIS平移去畸变
        odomDeskewInfo();

        return true;
    }

    // >IMU 去畸变
    // ! imu仅用于点云旋转去畸变
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;  

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // > get roll, pitch, and yaw estimation for this scan, 用于scan2map的初始化
            if (currentImuTime <= timeScanCur)
                //将message里面的IMU消息转为tf类型的数据，前者只是个float的类型的结构体 后者则是一个类 封装了很多函数
                // ! 将imu的朝向赋值给点云 ，如果非九轴imu在此处会不准
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            // 当前的IMU时间戳比lidar时间戳大过0.01s
            if (currentImuTime > timeScanNext + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime; //存储跟Lidar时间戳接近的IMU时间戳
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // > integrate rotation, 计算每一时刻imu相对于初始时刻imu(scan扫描起始时刻)的增量，
            // 计算时间差 当前的IMU时间戳减去上一时刻的IMU时间戳得到时间增量
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            // 手动累加过程，当前时刻的旋转等于上一时刻的旋转加上上一时刻角速度乘上时间增量
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;  // 索引值要减去个1

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    //  > 视觉里程计去畸变
    // ! VIS去畸变仅用于点云的平移去畸变，原版lio-sam是使用imu平移去畸变的，但是代码里把平移去畸变注释掉了
    // ?  vins的视觉里程计？？是的，它为LIS的scan2map和去畸变提供数据来源(除LIS初始化过程外)
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)  // ! 要求卡的很死，小于0.01s的误差
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)  // ! 这个操作是在找最接近timeScanCur的odomQueue[i]
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);   // 得到当前scan的起始时刻rpy

        // Initial guess used in mapOptimization
        // > 将点云信息中的位姿设置为视觉里程计的位姿
        cloudInfo.odomX = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll  = roll;
        cloudInfo.odomPitch = pitch;
        cloudInfo.odomYaw   = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;//表示此时视觉里程计可用

        // get end odometry at the end of the scan
        // 检查视觉里程计队列末尾的值
        odomDeskewFlag = false;

        // 如果视觉里程计队里末尾值的时间戳小于两帧lidar时间戳，说明视觉里程计频率过低，没有覆盖当前扫描帧
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        // VIS里程计末尾值
        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)  // 找最接近timeScanNext的VIS里程计
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        // 在visual_estimator/utility/visualization.cpp中发布给LIO视觉里程计信息的最后一项为failureCount
        // 初始值设为-1，每次clearState都会导致++failureCount
        // ! 如果前后的failureCount不一致说明在Lidar当前帧内视觉里程计至少重启了一次，跟踪失败，那么值就不准确了
        // 因此不使用视觉里程计对点云平移去畸变
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 扫描起始时刻全局位姿
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 扫描结束时刻全局位姿
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 扫描起止时刻之间的相对位姿变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        //在后续的findPosition中使用
        odomDeskewFlag = true;
    }

    // > 使用imu对点云旋转插值去畸变
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;

        //找到点云时间戳前最近的一个imu时间戳
        while (imuPointerFront < imuPointerCur)  // imuPointerCur是当前扫描时间内所覆盖的imu预积分数据，头尾Imu数据精准的把当前scan精准且密闭地夹住
        {
            //点云时间比IMU时间队列中最靠前的都小 则直接返回
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // ? 函数到这时imuPointerFront=imuPointerCur，也意思为这是最后一个扫描点
        // ? 最后一个imu数据其实不是要找的旋转量，为什么不插值呢，是因为没有覆盖到最后一个扫描点
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            //此时点云的姿态等于IMU的姿态
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } 
        else 
        { //如果点云时间还要靠前，则用插值的方法计算姿态
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    // > 在假设几乎匀速状态下，使用VIS里程计比例平移去畸变，原版说速度比较慢时不起太大作用，也注释掉了
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;
        //同样是按照差值的方式计算点云此时的位置
        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    // > 对点云中每个点进行去畸变,需要转到最开始的lidar位置
    // 由于载体存在移动，扫描一圈后事实上并不会形成圆形，但此时的xyz是相对于当前时刻的lidar，
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)  // 不去畸变
            return *point;

        //点云起始时间戳+相对于第一帧的时间戳
        double pointTime = timeScanCur + relTime;
        float rotXCur, rotYCur, rotZCur;

        //  ? 原注释：对点云位置去畸变 实际上计算的相对于世界坐标系原点的变化,
        // ? 原注释是不对的，这显然求出来是相对于imuQueue中第一个Imu系的坐标，见line536左右
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        //3.1.2对点云位置去畸变
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true) // ! firstPointFlag初始化为true，因为在去畸变时是按点云索引循环的，第一个索引就是对应第一个扫描点
        {
            // ! 这里得到的第一个扫描点的坐标是相对于imuQueue中第一个Imu系的坐标，但是我们现在需要做的工作是将所有的扫描点都转换到第一个扫描点上，而不是imuQueue中第一个Imu系的坐标 -> 所以此处求逆
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);

        // ! 当前扫描点相对于第一个扫描点的变换
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 以下相乘当前扫描点变换到第一个扫描点的坐标系下
        // [T00, T01, T02, T03]
        // [T10, T11, T12, T03]
        // [T20, T21, T22, T03]
        // [0.,  0.,  0.,  1. ]
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // > 将深度信息投影到RangeImage上
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)  // ! downsampleRate默认1，这里相当于对RI进行一次降采样
                continue;

            // ? 特殊的索引变换
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2; 
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            float range = pointDistance(thisPoint);

            if (range < 1.0)  // 小于1m不要
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) // 重复填充
                continue;

            // for the amsterdam dataset
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 点云去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);  //  ! Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0);  // !  Ouster

            int index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    // > 提取部分点云，主要是忽略RI的边缘点云
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry

        for (int i = 0; i < N_SCAN; ++i)//竖直方向
        {
            //从第一个scan开始
            cloudInfo.startRingIndex[i] = count - 1 + 5;//最开始的五个不考虑

            for (int j = 0; j < Horizon_SCAN; ++j)//水平方向
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;//最末尾的五个不考虑
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader; //点云的头部信息 包含时间戳、坐标系信息

         // > sensor_msgs/PointCloud2形式直接赋值
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}
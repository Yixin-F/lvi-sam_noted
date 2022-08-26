#include "utility.h"

#include <gtsam/geometry/Rot3.h>  // 四元数
#include <gtsam/geometry/Pose3.h> // 平移3X1
#include <gtsam/slam/PriorFactor.h> // 先验
#include <gtsam/slam/BetweenFactor.h> // 两结点间约束
#include <gtsam/navigation/GPSFactor.h> // gps因子
#include <gtsam/navigation/ImuFactor.h> // imu因子
#include <gtsam/navigation/CombinedImuFactor.h> // ?
#include <gtsam/nonlinear/NonlinearFactorGraph.h> // 非线性因子图，主体
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h> // L-M优化
#include <gtsam/nonlinear/Marginals.h> // 边缘化
#include <gtsam/nonlinear/Values.h> // 结点
#include <gtsam/inference/Symbol.h> // ?

#include <gtsam/nonlinear/ISAM2.h> // isam2优化器
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h> // ?

// > 优化三个变量，位姿、速度、Imu的Bias
using gtsam::symbol_shorthand::X; //  > Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // >  Vel   (xdot,ydot,zdot)  // ? 为什么要优化速度？在failureDetection()中体现为检测是否imu速度过大导致此次优化无效
using gtsam::symbol_shorthand::B; //  > Bias  (ax,ay,az,gx,gy,gz)

// IMU预积分的构造函数，继承自ParamServer类
class IMUPreintegration : public ParamServer
{
public:
    // 订阅和发布
    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    // map -> odom
    tf::Transform map_to_odom;
    tf::TransformBroadcaster tfMap2Odom;
    // odom -> base_link
    tf::TransformBroadcaster tfOdom2BaseLink;

    bool systemInitialized = false;

    // 噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::Vector noiseModelBetweenBias;

    // imu 预积分器
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // ! ----------------------------
    // > 两个 imu 数据队列, 
    // > imuQueOpt一直进行普通的高频率Imu帧之间的预积分，最原始的数据提供，参与优化过程
    // > imuQueImu得到被优化后的bias后，用最新的bias重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻最准的imu位姿
    // !----------------------------
    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;
    
    // imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;   //  Navigation state: Pose (rotation, translation) + velocity

    gtsam::imuBias::ConstantBias prevBias_;   // imu的bias也是待优化的变量
    
    // imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    // > gtsam优化三要素
    gtsam::ISAM2 optimizer;  //  isam2优化器
    gtsam::NonlinearFactorGraph graphFactors;  // 因子图
    gtsam::Values graphValues;  // 结点

    const double delta_t = 0;

    int key = 1;
    int imuPreintegrationResetId = 0;

    // imu-lidar位姿变换
     // > gtsam::Pose3 = gtsam::Rot3(四元数) + gtsam::Point3(xyz)
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));;

    // 构造函数
    IMUPreintegration()
    {
        // 订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预积分量，预测每一个时刻（imu频率）的imu里程计
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());  // > this 调用类内回调函数
        // 订阅激光里程计，来自mapOptimization，用两帧之间的imu预计分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
        subOdometry = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布imu 里程计
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> ("odometry/imu", 2000);
        pubImuPath     = nh.advertise<nav_msgs::Path>     (PROJECT_NAME + "/lidar/imu/path", 1);

        // >  tf::Transform = tf::createQuaternionFromRPY(, , ,)(四元数) + tf::Vector3(xyz)
        map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));

        // imu 预积分的噪声协方差
        // > 协方差的初值一般都是标定出来的，然后在后面的传播中更新，如isam2优化、滤波等
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);  // ? 重力去除，难道不用重力对齐z方向吗？
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous // ? 对角线形式，xyz互不影响？就是这样设计的，后面会传播
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias  // ? 初始bias不是标定来的吗，这里全默认为0了，直接等待优化？

        // !  噪声先验，初值取的没有道理，作者经过实验得到的
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        // 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // meter

        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        // imu 预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系了，与激光里程计同一个坐标系）
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        // imu 预积分器，用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    // 重置ISAM2 优化器
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);  // reset

        gtsam::NonlinearFactorGraph newGraphFactors;  // reset
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;  // reset
        graphValues = NewGraphValues;
    }

    // 重置参数
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /**
     * 订阅激光里程计，来自mapOptimization
     * 1、每隔100帧激光里程计，重置ISAM2优化器，添加里程计、速度、偏置先验因子，执行优化
     * 2、计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
     * 3、优化之后，执行重传播；优化更新了imu的偏置，用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
     *  // > 换言之，imu的频率高于激光帧频率，以k时刻已优化激光帧为准，在k+1时刻未优化激光帧到来之前，imu一直在预积分；
     *  // > 直到k+1时刻激光帧来到时，可以计算得到“基于k时刻已优化激光帧的，在k+1时刻的imu预积分量”+ "基于k时刻已优化激光帧的，在k+1时刻的未优化激光帧" -> isam2优化  == "imu在k+1时刻的优化的bias" + "k+1时刻的优化的激光帧"
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // 当前激光里程计时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        // 确保imu优化队列中有imu数据进行预积分
        if (imuQueOpt.empty())
            return;

        // 当前帧激光位姿，来自scan-to-map匹配，因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        int currentResetId = round(odomMsg->pose.covariance[0]);  // ? resetId是什么？
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // correction pose jumped, reset imu pre-integration
        if (currentResetId != imuPreintegrationResetId)
        {
            resetParams();
            imuPreintegrationResetId = currentResetId;  // ? 必须以某个激光帧为起始时刻进行预积分??
            return;
        }


        // 0. initialize system
        // 0. 系统初始化，第一帧
        if (systemInitialized == false)
        {
            // 重置ISAM2优化器
            resetOptimization();

            // pop old IMU message
            // 从imu优化队列中删除当前激光里程计时刻之前的imu数据
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }

            // ! -----------------------------------------------------------
            // > 以下是一个isam2初始化过程！！！！
            // ! -----------------------------------------------------------

            // > 添加因子使用graphFactors.add(factor)，至于添加在哪个位置在构建factor时已经参数输入
            // > 先验因子输入参数：位置类型，状态类型，相应噪声协方差
            // initial pose
            // 添加里程计位姿先验因子
            prevPose_ = lidarPose.compose(lidar2Imu); // !  将当前里程计位姿转换到imu坐标系下
            // >  X(0)表示第一个位姿，有一个先验的约束，约束内容为：lidar在imu下的prevPose_这么一个位姿
            // 该约束的权重，置信度为priorPoseNoise，越小代表置信度越高
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            // 约束添加到因子图当中
            graphFactors.add(priorPose);

            // initial velocity
            // > 添加速度先验因子，初始速度设为0，注意priorVelNoise非常大
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);

            // initial bias
            // ! 添加imu bias 先验因子，初值为零
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);

            // add values
            //  > 变量节点赋初值，使用graphValues(位置类型，状态量)
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // optimize once
            // 进行优化一次
            optimizer.update(graphFactors, graphValues);

            // > 送入优化器后保存约束和状态的变量就清零
            graphFactors.resize(0);
            graphValues.clear();

            // >  重置优化之后的bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;  // 代表优化了一帧，大于100帧时重置isam2优化器

            // 系统已经进行了初始化的标志
            systemInitialized = true;

            return;
        }


        // reset graph for speed
        // 每隔100帧激光里程计，重置ISAM2优化器，保证优化效率
        // ? 这样之前的检测到回环怎么办，因子图已经丢了？
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));

            // reset graph
            resetOptimization();

            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);

            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);

            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);

            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // ! --------------------------------------------------------------------
        // > 以下是一个正常优化过程(除了优化器的初始化和重置)
        // ! --------------------------------------------------------------------

        // 1. integrate imu data and optimize
        // Step 1 计算前一帧与当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，等待添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        while (!imuQueOpt.empty())  // >只要 imuQueOpt不空，并且提前于下一时刻激光帧到来，就进行积分
        {
            // pop and integrate imu data that is between two optimizations
            // 提取前一帧和当前帧之间的imu数据，计算预积分
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // 两帧imu数据时间间隔，包含了一个小的初始化"lastImuT_opt = -1"
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);

                // > imu预积分输入：加速度，角速度，dt
                // >  gtsam会自动完成预积分量的更新及协方差的更新等操作
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;

                // 队列中删除已经处理过的imu数据
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // add  preint imu factor to graph
        // >  添加imu预积分因子，这里指预积分量
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        //  > 参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一阵bias，预积分量
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);

        // add imu bias between factor
        // > 添加imu偏置因子，前一帧偏置，当前帧偏置，观测值，噪声协方差；deltaTij()是积分段的时间
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

        // add pose factor
        //  > 添加位姿因子，这里的位姿因子就是指激光帧计算得到位姿
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // 对于 LIO-SAM ，如果退化就让置信度小一些
        // LIO-SAM 此处代码为：
        // gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);  // ? 为什么仍然使用PriorFactor，难道没有其他因子类型给激光帧的位姿了吗？？
        graphFactors.add(pose_factor);

        // insert predicted values
        // > 预积分器用前一帧的状态和bias，施加imu预积分量，得到当前帧的状态
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);

        // >  imu预测状态的三个变量插入到因子图当中
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);

        // optimize
        //  > 优化两次
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();

        // Overwrite the beginning of the preintegration for the next step.
        //  > 优化结果，获取优化后的当前状态作为当前帧的最佳估计
        gtsam::Values result = optimizer.calculateEstimate();

        // 更新当前帧位姿，速度
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));

        // 更新当前帧状态
        prevState_ = gtsam::NavState(prevPose_, prevVel_);

        // 更新当前帧imu bias
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));

        // Reset the optimization preintegration object.
        // > 预积分器复位，设置新的偏置
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        // check optimization
        // > 检查imu因子图优化结果，如果速度或者bias过大，认为其失败
        if (failureDetection(prevVel_, prevBias_))
        {
            // 失败后重置参数
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // Step 2 优化之后，执行重传播；优化更新了imu的偏置，用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;

        // first pop imu message older than current correction data
        // 从 imu 队列中删除当前激光里程计时刻之前的imu数据
        // ? 也就是说，删除掉imuQueImu在当前激光帧之前的数据，如果imuQueImu没有覆盖下一次激光帧来的时候，那不会muQueImu积分量相对于下一次激光帧时刻残缺(下次积分时也删除了)一块？？
        // ? 除非imuQueImu只单纯用于imu路径的发布，轨迹可视化，只要保证从当前以光帧开始到下一激光帧来临之前的轨迹预测显示
        // ? 以上想法是错误的，首先要知道Imu肯定比激光帧来的快，这里是把相对于当前时刻激光帧优化等待多时的Imu数据积分掉，这时imuHandler()收到"doneFirstOpt = true"的命令把imuQueImu实时积分掉了，这里就没用了
        // ? 这里仅是一个把等了太久的imu积分掉而已，imuHandler()在苦等一个"初次优化成功的标志"，毕竟有failureDetection()牵制着
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }

        // repropogate
        // 对剩余的imu数据计算预积分
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // > 用最近的bias来重置预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 计算预积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    /**
     * @brief   检测一次isam2优化结果的有效性
     * 
     * @param[in] velCur 
     * @param[in] biasCur 
     * @return true 
     * @return false 
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 0.1 || bg.norm() > 0.1)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /**
     * 订阅imu原始数据
     * 1、用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态，也就是imu里程计
     * 2、imu里程计位姿转到lidar系，发布里程计
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        
        // imu原始测量数据转换到lidar系（加速度，角速度，RPY）
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // publish static tf
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, thisImu.header.stamp, "map", "odom"));
        // 添加当前帧imu数据到队列
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // >  要求上一次imu因子图优化执行成功，确保更新了上一帧（激光里程计帧）的状态和bias，这也是有且仅有一次的初始化过程
        // > 只要一次初始化成功了，odometryHandler()中 imuIntegratorImu_就不再预积分了，因为imuQueImu中的Imu都被实时积分掉了呀，没有剩下的！！
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        // 更新当前imu数据和上次imu数据的时间间隔dt，如果是第一次就认为 dt = 1 / 500.0 
        
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        //  > imu预积分器添加一帧imu数据，这个预积分的起始时刻是上一帧激光里程计时刻
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 用上一帧激光里程计时刻对应的状态和bias，施加从该时刻开始到当前时刻的imu预积分量，得到当前时刻的状态
        // 这里的值是优化过后的prevState
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        // 发布imu里程计（注意与激光里程计同一个系）
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // ! 变换到lidar坐标系
        // ! 因为是通过Imu预积分来预测下一次状态的，所以一开始把Lidar转换到imu系下，进行Isam2优化；但是发布imu预测轨迹里程计，需要转换到Lidar坐标系
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);  // imuPose在lidar坐标系下的pose，这个名字起得不太好 :)

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        
        // ! information for VINS initialization, 原始vins的初始化需要三角化，而现在只需要使用lio-sam的imu预积分结果里程计就可以
        odometry.pose.covariance[0] = double(imuPreintegrationResetId);
        odometry.pose.covariance[1] = prevBiasOdom.accelerometer().x();
        odometry.pose.covariance[2] = prevBiasOdom.accelerometer().y();
        odometry.pose.covariance[3] = prevBiasOdom.accelerometer().z();
        odometry.pose.covariance[4] = prevBiasOdom.gyroscope().x();
        odometry.pose.covariance[5] = prevBiasOdom.gyroscope().y();
        odometry.pose.covariance[6] = prevBiasOdom.gyroscope().z();
        odometry.pose.covariance[7] = imuGravity;
        pubImuOdometry.publish(odometry);

        // publish imu path
        // ? 这个imu path是给mapOptmization.cpp中的scan2map做初值用的吗？不是的，这里的path只是用来rviz显示
        // ? scan2map做初值首选VIS里程计提供，如果VIS失效，则直接由九轴imu提供，这些数据都存在cloud_info.msg里
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = thisImu.header.stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = odometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && abs(imuPath.poses.front().header.stamp.toSec() - imuPath.poses.back().header.stamp.toSec()) > 3.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = thisImu.header.stamp;
                imuPath.header.frame_id = "odom";
                pubImuPath.publish(imuPath);
            }
        }

        // publish transformation
        tf::Transform tCur;
        tf::poseMsgToTF(odometry.pose.pose, tCur);
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, thisImu.header.stamp, "odom", "base_link");
        tfOdom2BaseLink.sendTransform(odom_2_baselink);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");
    
    IMUPreintegration ImuP;

    ROS_INFO("\033[1;32m----> Lidar IMU Preintegration Started.\033[0m");

    ros::spin();
    
    return 0;
}
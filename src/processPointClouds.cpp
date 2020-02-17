// PCL lib Functions for processing point clouds

#include "processPointClouds.h"
#include <unordered_set>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>

//constructor:
template <typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

//de-constructor:
template <typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    typename pcl::PointCloud<PointT>::Ptr downsampled_cloud(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> downsampler;
    downsampler.setInputCloud(cloud);
    downsampler.setLeafSize(filterRes, filterRes, filterRes);
    downsampler.filter(*downsampled_cloud);

    typename pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>());
    pcl::CropBox<PointT> cropper;
    cropper.setMin(minPoint);
    cropper.setMax(maxPoint);
    cropper.setInputCloud(downsampled_cloud);
    cropper.filter(*filtered_cloud);

    pcl::CropBox<PointT> roof_cropper;
    roof_cropper.setMin(Eigen::Vector4f(-1.5, -2.0, -2.0, 1));
    roof_cropper.setMax(Eigen::Vector4f(3, 2.0, 1.0, 1));
    roof_cropper.setNegative(true);
    roof_cropper.setInputCloud(filtered_cloud);
    typename pcl::PointCloud<PointT>::Ptr roof_filtered_cloud(new pcl::PointCloud<PointT>());
    roof_cropper.filter(*roof_filtered_cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;
    return roof_filtered_cloud;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    pcl::ExtractIndices<PointT> extract;
    typename pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr outlier_cloud(new pcl::PointCloud<PointT>());
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*inlier_cloud);

    extract.setNegative(true);
    extract.filter(*outlier_cloud);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(outlier_cloud, inlier_cloud);
    return segResult;
}

template <typename PointT>
std::vector<int> PlaneRansac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{

    // Time ransac
    auto startTime = std::chrono::steady_clock::now();

    srand(time(NULL));

    std::unordered_set<int> inliersResult;
    const size_t num_points = cloud->points.size();
    while (maxIterations--)
    {
        std::unordered_set<int> inliers;
        while (inliers.size() < 3)
        {
            inliers.insert(rand() % num_points);
        }
        auto it = inliers.begin();
        const PointT &p1 = cloud->points[*(it++)];
        const PointT &p2 = cloud->points[*(it++)];
        const PointT &p3 = cloud->points[*(it++)];

        const float A = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
        const float B = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
        const float C = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
        const float D = -(A * p1.x + B * p1.y + C * p1.z);
        const float N = std::sqrt(A * A + B * B + C * C);

        for (int j = 0; j < num_points; ++j)
        {
            const PointT &p = cloud->points[j];
            const float d = std::abs(A * p.x + B * p.y + C * p.z + D) / N;
            if (d < distanceTol)
            {
                inliers.insert(j);
            }
        }
        //std::cout << "Iteration " << i << " resulted in " << inliers.size() << " of " << num_points << " inliers." << std::endl;
        if (inliers.size() > inliersResult.size())
        {
            std::swap(inliersResult, inliers);
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane ransac took " << elapsedTime.count() << " milliseconds" << std::endl;
    std::vector<int> indices;
    std::copy(inliersResult.begin(), inliersResult.end(), std::back_inserter(indices));
    return indices;
}

template <typename PointT>
std::vector<int> PlaneRansacEigen(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{

    // Time ransac
    auto startTime = std::chrono::steady_clock::now();

    srand(time(NULL));

    std::vector<int> inliers;
    const size_t num_points = cloud->points.size();
    std::cout << "num_points: " << num_points << std::endl;

    Eigen::ArrayXf best_inliers;
    size_t num_best_inliers = 0;
    float best_distanceTolN;

    while (maxIterations--)
    {
        std::unordered_set<int> samples;
        while (samples.size() < 3)
        {
            const int point_index = rand() % num_points;
            samples.insert(point_index);
        }
        auto it = samples.begin();
        const PointT &p1 = cloud->points[*(it++)];
        const PointT &p2 = cloud->points[*(it++)];
        const PointT &p3 = cloud->points[*(it++)];

        const float A = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
        const float B = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
        const float C = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
        const float D = -(A * p1.x + B * p1.y + C * p1.z);
        const float N = std::sqrt(A * A + B * B + C * C);
        const float distanceTolN = distanceTol * N;
        Eigen::MatrixXf point_mat(num_points, 4);
        for (int j = 0; j < num_points; ++j)
        {
            const PointT &p = cloud->points[j];
            point_mat.row(j) << p.x, p.y, p.z, 1.f;
        }
        //std::cout << "point mat : \n" << point_mat << std::endl;
        const Eigen::Vector4f plane(A, B, C, D);
        const Eigen::ArrayXf point_distancesN = (point_mat * plane);
        //std::cout << "pont d : \n" << point_distancesN.abs() << " > " << distanceTolN << std::endl;

        size_t num_sample_inliers = (point_distancesN.abs() < distanceTolN).count();
        if (num_sample_inliers > num_best_inliers)
        {
            best_inliers = point_distancesN;
            best_distanceTolN = distanceTolN;
            num_best_inliers = num_sample_inliers;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane ransac took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::vector<int> inliners;

    if (num_best_inliers > 0)
    {
        for (int j = 0; j < num_points; ++j)
        {
            if (best_inliers(j, 0) < best_distanceTolN)
                inliers.push_back(j);
        }
    }
    return inliers;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    auto startTime = std::chrono::steady_clock::now();
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    inliers->indices = PlaneRansac<PointT>(cloud, maxIterations, distanceThreshold);
    std::cout << "Plane segmentation resulted in " << inliers->indices.size() << " inliers " << std::endl;

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlanePCL(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    typename pcl::PointCloud<PointT>::Ptr downsampled_cloud(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> downsampler;
    downsampler.setInputCloud(cloud);
    downsampler.setLeafSize(0.1f, 0.1f, 0.1f);
    downsampler.filter(*downsampled_cloud);
    std::cout << "Cloud decimated to " << downsampled_cloud->points.size() << " from " << cloud->points.size() << std::endl;
    pcl::SACSegmentation<PointT> segmentation;
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setMaxIterations(maxIterations);
    segmentation.setDistanceThreshold(distanceThreshold);

    segmentation.setInputCloud(downsampled_cloud);

    pcl::ModelCoefficients coefficients;
    segmentation.segment(*inliers, coefficients);

    std::cout << "Plane segmentation resulted in " << inliers->indices.size() << " inliers with plane parameters " << coefficients << std::endl;
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, downsampled_cloud);
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "pcl plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    return segResult;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    for (const auto &indices : cluster_indices)
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        cloud_cluster->points.resize(indices.indices.size());
        size_t j = 0;
        for (auto i : indices.indices)
        {
            cloud_cluster->points[j++] = cloud->points[i];
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters.push_back(cloud_cluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}
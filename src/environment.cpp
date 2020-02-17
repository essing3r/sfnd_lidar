/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "sensors/lidar.h"
#include "render/render.h"
#include "processPointClouds.h"
#include <pcl/point_types.h>
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

std::vector<Car> initHighway(bool renderScene, pcl::visualization::PCLVisualizer::Ptr &viewer)
{

    Car egoCar(Vect3(0, 0, 0), Vect3(4, 2, 2), Color(0, 1, 0), "egoCar");
    Car car1(Vect3(15, 0, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car1");
    Car car2(Vect3(8, -4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car2");
    Car car3(Vect3(-12, 4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car3");

    std::vector<Car> cars;
    cars.push_back(egoCar);
    cars.push_back(car1);
    cars.push_back(car2);
    cars.push_back(car3);

    if (renderScene)
    {
        renderHighway(viewer);
        egoCar.render(viewer);
        car1.render(viewer);
        car2.render(viewer);
        car3.render(viewer);
    }

    return cars;
}

template<typename PointT>
void cityBlock(pcl::visualization::PCLVisualizer::Ptr &viewer, ProcessPointClouds<PointT>* processor, typename pcl::PointCloud<PointT>::Ptr& cloud)
{
    typename pcl::PointCloud<PointT>::Ptr filtered_cloud = processor->FilterCloud(cloud, 0.1f, Eigen::Vector4f (-10.0, -4.0, -4.0, 1), Eigen::Vector4f (30.0, 7.0, 4.0, 1));
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> obstacle_ground_pointclouds;
    obstacle_ground_pointclouds = processor->SegmentPlane(filtered_cloud, 30, 0.2f);
    //renderPointCloud(viewer, obstacle_ground_pointclouds.first, "obstacles", Color(1,0,0));
    renderPointCloud(viewer, obstacle_ground_pointclouds.second, "ground", Color(0, 1, 0));
    
    std::vector<typename pcl::PointCloud<PointT>::Ptr> cluster_clouds = processor->ClusteringNoPCL(obstacle_ground_pointclouds.first, 0.4, 10, 5000);
    std::vector<Color> colors = {Color(1., 0, 0), Color(1, 1, 0), Color(0, 0, 1)};
    int cluster_id = 0;
    for (const auto &cluster : cluster_clouds)
    {
        std::cout << "cluster size ";
        processor->numPoints(cluster);
        renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(cluster_id), colors[cluster_id % colors.size()]);
        Box box = processor->BoundingBox(cluster);
        renderBox(viewer, box, cluster_id);
        cluster_id++;
    }
}

void simpleHighway(pcl::visualization::PCLVisualizer::Ptr &viewer)
{
    // ----------------------------------------------------
    // -----Open 3D viewer and display simple highway -----
    // ----------------------------------------------------

    // RENDER OPTIONS
    bool renderScene = false;
    std::vector<Car> cars = initHighway(renderScene, viewer);

    // TODO:: Create lidar sensor
    Lidar::Ptr lidar = std::make_shared<Lidar>(cars, 0.);

    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud = lidar->scan();
    //renderRays(viewer, lidar->position, lidar_cloud);
    //renderPointCloud(viewer, lidar_cloud, "lidar_cloud");
    // TODO:: Create point processor
    ProcessPointClouds<pcl::PointXYZ> processor;
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> obstacle_ground_pointclouds;
    obstacle_ground_pointclouds = processor.SegmentPlane(lidar_cloud, 1000, 0.2f);
    //renderPointCloud(viewer, obstacle_ground_pointclouds.first, "obstacles", Color(1,0,0));
    renderPointCloud(viewer, obstacle_ground_pointclouds.second, "ground", Color(0, 1, 0));

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_clouds = processor.Clustering(obstacle_ground_pointclouds.first, 1, 1, 1000);
    std::vector<Color> colors = {Color(1., 0, 0), Color(1, 1, 0), Color(0, 0, 1)};
    int cluster_id = 0;
    for (const auto &cluster : cluster_clouds)
    {
        std::cout << "cluster size ";
        processor.numPoints(cluster);
        renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(cluster_id), colors[cluster_id % colors.size()]);
        Box box = processor.BoundingBox(cluster);
        renderBox(viewer, box, cluster_id);
        cluster_id++;
    }
}

//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr &viewer)
{

    viewer->setBackgroundColor(0, 0, 0);

    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;

    switch (setAngle)
    {
    case XY:
        viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0);
        break;
    case TopDown:
        viewer->setCameraPosition(0, 0, distance, 1, 0, 1);
        break;
    case Side:
        viewer->setCameraPosition(0, -distance, 0, 0, 0, 1);
        break;
    case FPS:
        viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if (setAngle != FPS)
        viewer->addCoordinateSystem(1.0);
}

int main(int argc, char **argv)
{
    std::cout << "starting enviroment" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    CameraAngle setAngle = XY;
    initCamera(setAngle, viewer);
    simpleHighway(viewer);

    ProcessPointClouds<pcl::PointXYZI> *pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();
    std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("../src/sensors/data/pcd/data_1");
    auto streamIterator = stream.begin();
    pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;
    while (!viewer->wasStopped())
    {

        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        // Load pcd and run obstacle detection process
        inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
        cityBlock(viewer, pointProcessorI, inputCloudI);

        streamIterator++;
        if (streamIterator == stream.end())
            streamIterator = stream.begin();

        viewer->spinOnce();
    }
}
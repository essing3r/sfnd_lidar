/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	// Add inliers
	float scatter = 0.6;
	for (int i = -5; i < 5; i++)
	{
		double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		pcl::PointXYZ point;
		point.x = i + scatter * rx;
		point.y = i + scatter * ry;
		point.z = 0;

		cloud->points.push_back(point);
	}
	// Add outliers
	int numOutliers = 10;
	while (numOutliers--)
	{
		double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
		pcl::PointXYZ point;
		point.x = 5 * rx;
		point.y = 5 * ry;
		point.z = 0;

		cloud->points.push_back(point);
	}
	cloud->width = cloud->points.size();
	cloud->height = 1;

	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}

pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("2D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
	viewer->addCoordinateSystem(1.0);
	return viewer;
}

std::unordered_set<int> PlaneRansac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
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
		const pcl::PointXYZ &p1 = cloud->points[*(it++)];
		const pcl::PointXYZ &p2 = cloud->points[*(it++)];
		const pcl::PointXYZ &p3 = cloud->points[*(it++)];

		const float A = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
		const float B = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
		const float C = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
		const float D = -(A * p1.x + B * p1.y + C * p1.z);
		const float N = std::sqrt(A * A + B * B + C * C);

		for (int j = 0; j < num_points; ++j)
		{
			const pcl::PointXYZ &p = cloud->points[j];
			const float d = std::abs(A * p.x + B * p.y + C *p.z) / N;
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
	return inliersResult;
}

std::vector<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{

	// Time ransac
	auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	const size_t num_points = cloud->points.size();

	// TODO: Fill in this function
	for (int i = 0; i < maxIterations; ++i)
	{

		pcl::PointXYZ p1 = cloud->points[rand() % num_points];
		pcl::PointXYZ p2 = cloud->points[rand() % num_points];

		const float a = p1.y - p2.y;
		const float b = p2.x - p1.x;
		const float c = (p1.x * p2.y - p2.x * p1.y);
		const float n = std::sqrt(a * a + b * b);
		//std::cout << "Sample " << i << "parameters a,b,c " << a << ","<< b << "," << c << "," << n << std::endl;

		std::unordered_set<int> inliers;
		for (int j = 0; j < num_points; ++j)
		{
			const pcl::PointXYZ &p = cloud->points[j];
			const float d = std::abs(a * p.x + b * p.y + c) / n;
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
	std::cout << "ransac took " << elapsedTime.count() << " milliseconds" << std::endl;

	return indices;
}

int main()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = PlaneRansac(cloud, 20, 0.5);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for (int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if (inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}

	// Render 2D point cloud with inliers and outliers
	if (inliers.size())
	{
		renderPointCloud(viewer, cloudInliers, "inliers", Color(0, 1, 0));
		renderPointCloud(viewer, cloudOutliers, "outliers", Color(1, 0, 0));
	}
	else
	{
		renderPointCloud(viewer, cloud, "data");
	}

	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
	}
}

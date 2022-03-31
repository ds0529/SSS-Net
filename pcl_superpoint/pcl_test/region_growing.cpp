#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

using namespace std;

void getAllFiles2(string path, vector<string>& files, string fileType)
{
	// 文件句柄
	intptr_t hFile = 0;
	// 文件信息
	struct _finddata_t fileinfo;

	string p;

	if ((hFile = _findfirst(p.assign(path).append("\\*" + fileType).c_str(), &fileinfo)) != -1) {
		do {
			// 保存文件的全路径
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));

		} while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1

		_findclose(hFile);
	}
}

int main(int argc, char** argv)
{
	string dataset = "s3dis";
	vector<string> all_file_names;
	if (dataset == "s3dis")	getAllFiles2("F:\\datasets\\S3DIS\\input_0.040\\pcb_XYZ", all_file_names, ".pcb");
	else if (dataset == "scannet_train") getAllFiles2("F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZ\\train\\", all_file_names, ".pcb");
	else if (dataset == "scannet_test")	getAllFiles2("F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZ\\test\\", all_file_names, ".pcb");
	else if (dataset == "semantic3d")	getAllFiles2("F:\\datasets\\Semantic3D\\input_0.060\\pcb_XYZ\\", all_file_names, ".pcb");
	for (int i = 0; i < all_file_names.size(); ++i)
	{
		cout << all_file_names[i] << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile <pcl::PointXYZ>(all_file_names[i], *cloud) == -1)
		{
			std::cout << "Cloud reading failed." << std::endl;
			return (-1);
		}
		cout << "point num: " << cloud->height*cloud->width << endl;

		//normal estimate
		pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
		normal_estimator.setSearchMethod(tree);
		normal_estimator.setInputCloud(cloud);
		normal_estimator.setKSearch(10);//50
		normal_estimator.compute(*normals);

		//setting
		pcl::IndicesPtr indices(new std::vector <int>);
		pcl::PassThrough<pcl::PointXYZ> pass;
		pass.setInputCloud(cloud);
		//pass.setFilterFieldName("z");
		//pass.setFilterLimits(0.0, 1.0);
		pass.filter(*indices);
		pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
		reg.setMinClusterSize(10);//50
		//reg.setMaxClusterSize(1000000);
		reg.setSearchMethod(tree);
		reg.setNumberOfNeighbours(10);//30
		reg.setInputCloud(cloud);
		//reg.setIndices (indices);
		reg.setInputNormals(normals);
		reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);//3.0 / 180.0 * M_PI
		reg.setCurvatureThreshold(1.5);//1.0

		//segment
		std::vector <pcl::PointIndices> clusters;
		reg.extract(clusters);
		std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
		std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
		std::cout << "These are the indices of the points of the initial" <<
			std::endl << "cloud that belong to the first cluster:" << std::endl;

		/*pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
		pcl::visualization::CloudViewer viewer("Cluster viewer");
		viewer.showCloud(colored_cloud);
		while (!viewer.wasStopped())
		{
		}*/

		//output
		vector<int> point_cluster_index(cloud->height*cloud->width, -1);
		for (int i = 0; i < clusters.size(); i++) {
			for (int j = 0; j < clusters[i].indices.size(); j++) {
				point_cluster_index[clusters[i].indices[j]] = i;
			}
		}
		//获取不带路径的文件名
		string::size_type iPos = all_file_names[i].find_last_of('\\') + 1;
		string filename = all_file_names[i].substr(iPos, all_file_names[i].length() - iPos);
		cout << filename << endl;
		//获取不带后缀的文件名
		string name = filename.substr(0, filename.rfind("."));
		cout << name << endl;
		string output_file_name;
		if (dataset == "s3dis")	output_file_name = "F:\\datasets\\S3DIS\\input_0.040\\region_growing\\";
		else if (dataset == "scannet_train") output_file_name = "F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing\\train\\";
		else if (dataset == "scannet_test") output_file_name = "F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing\\test\\";
		else if (dataset == "semantic3d") output_file_name = "F:\\datasets\\Semantic3D\\input_0.060\\region_growing\\";
		output_file_name = output_file_name + name + ".txt";
		fstream of(output_file_name, fstream::out);
		for (int r = 0; r < point_cluster_index.size(); r++)
		{
			of << point_cluster_index[r] << "\n";    //每列数据用 tab 隔开
		}
		of.close();
	}

	return (0);
}
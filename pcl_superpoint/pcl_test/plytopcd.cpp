#include <io.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <pcl/visualization/cloud_viewer.h>  
#include <pcl/conversions.h>
using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

void getAllFiles1(string path, vector<string>& files, string fileType)
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

int main1()
{
	string dataset = "semantic3d";
	vector<string> all_file_names;
	if (dataset == "s3dis")	getAllFiles1("F:\\datasets\\S3DIS\\input_0.040", all_file_names, ".ply");
	else if (dataset == "scannet_train") getAllFiles1("F:\\datasets\\ScanNet\\scans\\input_0.020\\train", all_file_names, ".ply");
	else if (dataset == "scannet_test")	getAllFiles1("F:\\datasets\\ScanNet\\scans\\input_0.020\\test", all_file_names, ".ply");
	else if (dataset == "semantic3d")	getAllFiles1("F:\\datasets\\Semantic3D\\input_0.060", all_file_names, ".ply");
	for (int i = 0; i < all_file_names.size(); ++i)
	{
		cout << all_file_names[i] << endl;
		//获取不带路径的文件名
		string::size_type iPos = all_file_names[i].find_last_of('\\') + 1;
		string filename = all_file_names[i].substr(iPos, all_file_names[i].length() - iPos);
		cout << filename << endl;
		//获取不带后缀的文件名
		string name = filename.substr(0, filename.rfind("."));
		cout << name << endl;
		//ply to pcb
		pcl::PCLPointCloud2 original_point_cloud;
		pcl::PLYReader reader;
		reader.read(all_file_names[i], original_point_cloud);
		pcl::PointCloud<pcl::PointXYZ> point_cloud_XYZ;
		pcl::PointCloud<pcl::PointXYZRGB> point_cloud_XYZRGB;
		pcl::fromPCLPointCloud2(original_point_cloud, point_cloud_XYZ);
		pcl::fromPCLPointCloud2(original_point_cloud, point_cloud_XYZRGB);
		/*pcl::visualization::CloudViewer viewer("viewer");
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ptr = point_cloud_XYZRGB.makeShared();
		viewer.showCloud(cloud_ptr);
		while (!viewer.wasStopped())
		{
		}*/
		pcl::PCDWriter writer;
		string output_file_name_XYZ;
		string output_file_name_XYZRGB;
		if (dataset == "s3dis")	output_file_name_XYZ = "F:\\datasets\\S3DIS\\input_0.040\\pcb_XYZ\\";
		else if (dataset == "scannet_train") output_file_name_XYZ = "F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZ\\train\\";
		else if (dataset == "scannet_test") output_file_name_XYZ = "F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZ\\test\\";
		else if (dataset == "semantic3d")	output_file_name_XYZ = "F:\\datasets\\Semantic3D\\input_0.060\\pcb_XYZ\\";
		if (dataset == "s3dis")	output_file_name_XYZRGB = "F:\\datasets\\S3DIS\\input_0.040\\pcb_XYZRGB\\";
		else if (dataset == "scannet_train") output_file_name_XYZRGB = "F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZRGB\\train\\";
		else if (dataset == "scannet_test") output_file_name_XYZRGB = "F:\\datasets\\ScanNet\\scans\\input_0.020\\pcb_XYZRGB\\test\\";
		else if (dataset == "semantic3d")	output_file_name_XYZRGB = "F:\\datasets\\Semantic3D\\input_0.060\\pcb_XYZRGB\\";
		output_file_name_XYZ = output_file_name_XYZ + name + ".pcb";
		output_file_name_XYZRGB = output_file_name_XYZRGB + name + ".pcb";
		cout << output_file_name_XYZ << endl;
		cout << output_file_name_XYZRGB << endl;
		writer.writeASCII(output_file_name_XYZ, point_cloud_XYZ);
		writer.writeASCII(output_file_name_XYZRGB, point_cloud_XYZRGB);
	}

	return 0;
}
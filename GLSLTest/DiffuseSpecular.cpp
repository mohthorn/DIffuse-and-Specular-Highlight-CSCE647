// GLSLTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#define WIDTH 800
#define HEIGHT 800

using namespace cv;
using namespace glm;


double inCircle(double x, double y, double x0, double y0, double r)
{
	double d = (x - x0)*(x - x0) + (y - y0)*(y - y0) - r * r;
	if (d > 0)
		d = 0;
	//cout << sqrt(-d) / r <<endl;
	return sqrt(-d)/r;
}

Mat colorLift(Mat & original, double value)
{
	Mat ret = original.clone();
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++)
		{
			for (int p = 0; p < 3; p++)
			{
				double cVal;
				cVal = original.at<Vec3b>(Point(i, j))[p] + value;
				if (cVal > 255)
				{
					cVal = 255;
				}
				if (cVal < 0)
				{
					cVal = 0;
				}
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
		}
	}
	return ret;
}

Mat diffuse(Mat & original, Mat & normal, Mat & specular)
{
	Mat ret = original.clone();
	Mat light = colorLift(original, 0);
	Mat dark = colorLift(original, -255);
	imshow("light", light);
	imshow("dark", dark);
	double lightSource[3] = { 100,400,50 };
	for (int i = 0; i < original.rows; i++)
	{ 
		for (int j = 0; j < original.cols; j++)
		{
			vec3 L = normalize(vec3(lightSource[0] - i, lightSource[1] - j, lightSource[2]));
			double norm[3];
			for (int p = 0; p < 3; p++)
			{
				norm[p]=(double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2*norm[p] / 255.0 -1 );				
			}
			//cout << norm[2] << " " << (i-400)/300.0 << endl;
			//cout << norm[1] << " " << (j - 400) / 300.0 << endl;
			//cout << norm[0] << " " << 1 << endl;
			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double t = (dot(L, N) + 1) / 2.0;
			for (int p = 0; p < 3; p++)
			{
				double cVal = light.at<Vec3b>(Point(i, j))[p] * t + dark.at<Vec3b>(Point(i, j))[p] *(1-t);
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
			vec3 R = normalize(-1.0f * L + 2.0f *(dot(L, N)*N) );
			vec3 E = vec3(0, 0, 1);
			double s = 0.5 * dot(R, E) + 0.5;
			for (int p = 0; p < 3; p++)
			{
				double cVal = ret.at<Vec3b>(Point(i, j))[p] * (1-(0.9*s)) + specular.at<Vec3b>(Point(i, j))[p] * (0.9*s);
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
		}
	}

	
	return ret;
}

void circleDiffusion()
{
	Mat original(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	Mat depth(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	Mat normal(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255/2.0, 255/2.0));
	Mat specular(WIDTH, HEIGHT, CV_8UC3, Scalar(200, 200, 200));
	for (int i = 0; i < depth.rows; i++)
		for (int j = 0; j < depth.cols; j++)
		{
			double z = 0;
			if ((z = inCircle(i, j, 400, 400, 300)) != 0)
			{
				Vec3b color = depth.at<Vec3b>(Point(i, j));
				color[0] = z * 255;
				color[1] = z * 255;
				color[2] = z * 255;
				depth.at<Vec3b>(Point(i, j)) = color;
				original.at<Vec3b>(Point(i, j))[1] = 255;
				normal.at<Vec3b>(Point(i, j))[0] = 1*255;
				normal.at<Vec3b>(Point(i, j))[1] = ((j - 400)/ 300.0 +1)/2.0 * 255;
				normal.at<Vec3b>(Point(i, j))[2] = ((i - 400) / 300.0 + 1) / 2.0 * 255;
				specular.at<Vec3b>(Point(i, j))[0] = 255;
				specular.at<Vec3b>(Point(i, j))[1] = 255;
				specular.at<Vec3b>(Point(i, j))[2] = 255;
			}

		}
	imshow("original", original);
	//imshow("depth", depth);
	imshow("normal", normal);
	Mat dif=diffuse(original, normal,specular);
	imshow("dif", dif);
	waitKey(0); // Wait for a keystroke in the window
}

int main()
{
	circleDiffusion();
}



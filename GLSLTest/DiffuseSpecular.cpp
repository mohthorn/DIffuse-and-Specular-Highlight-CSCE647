// GLSLTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#define WIDTH 800
#define HEIGHT 800
#define Radiant 300.0
#define X 8 //for antialiasing

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

double jitteringCircle(double x, double y, double x0, double y0, double r)
{
	double sum = 0;
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-0.5, 0.5);

	double ri = dis(gen);
	double rj = dis(gen);

	for (int i = 0; i < X; i++)
	{
		for (int j = 0; j < X; j++)
		{
			double m = x + (i + 0.5)*1.0 / X + ri / X;  //jittering
			double n = y + (j + 0.5)*1.0 / X + rj / X;
			sum += inCircle(m, n, x0, y0, Radiant);
		}
	}
	//if (sum <= 0.0001)
	//	return 0;
	return sum * 0.9 / (X*X) + (X*X - sum)*0.1 / (X*X);
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

Mat diffuse(Mat & original, Mat & normal, Mat & specular, Mat &depth)
{
	Mat ret = original.clone();
	Mat light = colorLift(original, 100);
	Mat dark = colorLift(original, -200);
	imshow("light", light);
	imshow("dark", dark);
	double lightSource[3] = { 200,200,400 };
	for (int i = 0; i < original.rows; i++)
	{ 
		for (int j = 0; j < original.cols; j++)
		{
			vec3 L = normalize(vec3(lightSource[0] - i, lightSource[1] - j, lightSource[2]-depth.at<Vec3b>(Point(i, j))[1]/255.0*Radiant));
			double norm[3];
			for (int p = 0; p < 3; p++)
			{
				norm[p]=(double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2*norm[p] / 255.0 -1 );				
			}
			norm[0] = sqrt(1 - norm[1] * norm[1] - norm[2] * norm[2]);

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double t = ((double)dot(L, N) + 1) / 2.0;
			for (int p = 0; p < 3; p++)
			{
				double cVal = light.at<Vec3b>(Point(i, j))[p] * t + dark.at<Vec3b>(Point(i, j))[p] *(1-t);
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
			vec3 R = -1.0f * L + 2.0f *(dot(L, N)*N) ;
			vec3 E = vec3(0, 0, 1);
			double s = ((double)dot(R, E) + 1) / 2.0;
			double ks = 0.9;
			for (int p = 0; p < 3; p++)
			{
				double cVal = ret.at<Vec3b>(Point(i, j))[p] * (1-(ks*s)) + specular.at<Vec3b>(Point(i, j))[p] * (ks*s);
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
	Mat specular(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < depth.rows; i++)
		for (int j = 0; j < depth.cols; j++)
		{
			double z = 0;
			if ((z = jitteringCircle(i, j, 400, 400, Radiant)) >0.1)
			{
				Vec3b color = depth.at<Vec3b>(Point(i, j));
				z *= Radiant;
				color[0] = z/ Radiant * 255;
				color[1] = z/ Radiant * 255;
				color[2] = z/ Radiant * 255;
				depth.at<Vec3b>(Point(i, j)) = color;
				original.at<Vec3b>(Point(i, j))[1] = 255;
				vec3 colorVec = normalize(vec3(z, (j - 400) , (i - 400)));
				normal.at<Vec3b>(Point(i, j))[0] = (colorVec[0] + 1) / 2.0 * 255;
				normal.at<Vec3b>(Point(i, j))[1] = (colorVec[1]+1)/2.0*255;
				normal.at<Vec3b>(Point(i, j))[2] = (colorVec[2]+1) / 2.0 * 255;
				specular.at<Vec3b>(Point(i, j))[0] = 255;
				specular.at<Vec3b>(Point(i, j))[1] = 255;
				specular.at<Vec3b>(Point(i, j))[2] = 255;
			}

		}
	imshow("original", original);
	//imshow("depth", depth);
	imwrite("depth.png", depth);
	imshow("normal", normal);
	imwrite( "normal.png", normal);
	imshow("specular", specular);
	Mat dif=diffuse(original, normal,specular,depth);
	imshow("dif", dif);
	imwrite("result.png", dif);
	waitKey(0); // Wait for a keystroke in the window
}

int main()
{
	circleDiffusion();
}



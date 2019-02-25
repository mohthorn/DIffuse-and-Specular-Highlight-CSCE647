// HW1 2D Diffuse Shading
// Chengyi Min 127004714

#include "pch.h"
#define WIDTH 800
#define HEIGHT 800
#define Radiant 300.0
#define X 1 //for anti aliasing



Mat simpleBlur(Mat & original)
{
	int steps = 5;
	Mat ret = original.clone();
	for(int i=0;i <original.cols;i++)
		for (int j = 0; j < original.rows; j++)
		{
			vec3 color = {0,0,0};
			for(int m=0;m<5;m++)
				for (int n = 0; n < 5; n++)
				{
					int x_tmp = i + m - steps /2;
					int y_tmp = j + n - steps /2;
					x_tmp = x_tmp % original.cols;
					y_tmp =y_tmp%original.rows;
					if (x_tmp < 0)
						x_tmp = 0;
					if (y_tmp < 0)
						y_tmp = 0;
					color = color + vec3(original.at<Vec3b>(Point(x_tmp, y_tmp))[0],
										original.at<Vec3b>(Point(x_tmp, y_tmp))[1],
										original.at<Vec3b>(Point(x_tmp, y_tmp))[2]);
				}
			for (int p = 0; p < 3; p++)
			{
				ret.at<Vec3b>(Point(i, j))[p] = color[p] / (double)(steps*steps);
			}
		}
	return ret;
}


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
			sum += inCircle(m, n, x0, y0, r);
		}
	}
	//if (sum <= 0.0001)
	//	return 0;
	return sum * 0.9 / (X*X) + (X*X - sum)*0.1 / (X*X);
}



Mat colorLift(Mat & original, double value,int channelStart, int channelEnd)
{
	Mat ret = original.clone();
	for (int i = 0; i < original.cols; i++)
	{
		for (int j = 0; j < original.rows; j++)
		{
			for (int p = channelStart; p < channelEnd; p++)
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


Mat fresnel(Mat & normal, Mat & refl, Mat & refr, int zAdjust=0)
{
	Mat ret = refl.clone();

	vec3 E = vec3(0, 0, 1);
	double norm[3];
	//double real_d = dist - (double)depth.at<Vec3b>(Point(i, j))[0];

	double x0 =-0.0001,y0=0.1, x1=0.2, x2=0.8;



	for(int i=0; i < normal.cols;i++)
		for (int j = 0; j < normal.rows; j++)
		{
			double F = 0;
			for (int p = 0; p < 3; p++)
			{
				norm[p] = (double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2.0 * norm[p] / 255.0 - 1);
			}
			if (zAdjust)
				norm[0] = (double)zAdjust / 100.0;

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double c = (1 - dot(N, E) );
			if(c>x0 && c<=x1)
			{
				F = (0-y0)/(x1-x0)*(c-x0)+y0;

			}
			if (c > x1&&c <= x2)
			{
				F = (1 - 0) / (x2 - x1)*(c-x1) + 0;
			}
			if (c>x2)
			{
				F = 1;
			}
			for (int p = 0; p < 3; p++)
			{
				ret.at<Vec3b>(Point(i, j))[p] = (1 - F)*(double)refr.at<Vec3b>(Point(i, j))[p] + F * (double)refl.at<Vec3b>(Point(i, j))[p];
			}
		}
	return ret;
}

Mat refraction(double dist, Mat & original, Mat & normal, Mat & bg, Mat &depth, double yeta, int rough =0,int zAdjust = 0)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-0.05, 0.05);
	Mat ret = original.clone();
	int env_width = bg.cols;
	int env_height = bg.rows;
	double yibu = log2(yeta);
	int steps = 1;
	if (rough)
	{
		steps = rough;
	}
	for (int i = 0; i < original.cols; i++)
	{
		for (int j = 0; j < original.rows; j++)
		{
			double color[3] = { 0,0,0 };
			vec3 E = vec3(0, 0, 1);
			double norm[3];
			double real_d = dist - (double)depth.at<Vec3b>(Point(i, j))[0];

			for (int p = 0; p < 3; p++)
			{
				norm[p] = (double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2.0 * norm[p] / 255.0 - 1);
			}
			if (zAdjust)
				norm[0] = (double)zAdjust / 100.0;

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			for (int m = 0; m < steps; m++)
				for (int n = 0; n < steps; n++)
				{
					vec3 s = vec3(0,0,0);
					if (rough)
					{
						double s1 = dis(gen);
						double s2 = dis(gen);
						double s3 = dis(gen);
						s = vec3(s1, s2, s3);
						//N = normalize(N + vec3(s1, s2, s3));
					}

					vec3 T = vec3(0, 0, -1);
					if (yibu > 0)
						T = (float)(yibu)*(-N) + (float)(1.0f - yibu) *(-E);
					else if (yibu < 0)
					{
						vec3 M = E - dot(E, N) * N;
						T = (float)(yibu)*(M)+(float)(1.0f + yibu) *(-E);
					}
					T = T + s;
					//printf("%f,%f,%f\n", Re[0], Re[1], Re[2]);
					float scale = (float)(fabs(real_d) / fabs((double)T[2] + 0.0001));

					int x = i + T[0] * scale + env_width / 2 - original.cols / 2;
					int y = j + T[1] * scale + env_height / 2 - original.rows / 2;
					x = x % env_width;
					y = y % env_height;
					if (x < 0)
						x = -x;

					if (y < 0)
						y = -y;

					for (int p = 0; p < 3; p++)
					{
						color[p] = color[p] + (double)bg.at<Vec3b>(Point(x, y))[p];
					}
				}
			double ks = 0.8;
			for (int p = 0; p < 3; p++)
			{
				ret.at<Vec3b>(Point(i, j))[p] = (1 - ks)*original.at<Vec3b>(Point(i, j))[p] + ks * color[p] / (double)(steps*steps);
			}
		}
	}

	//if (rough)
	//	ret = simpleBlur(ret);
	return ret;
}



Mat reflection(double dist, Mat & original, Mat & normal, Mat & env, Mat &depth, int glossy, int zAdjust = 0)
{
	Mat ret = original.clone();
	int env_width = env.cols;
	int env_height = env.rows;
	printf("environment map: %d * %d\n", env_width, env_height);
	double d = dist;
	
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-0.1,0.1);
	double r_exp = 1;
	int steps = 1;
	if (glossy)
		steps = glossy;
	for (int i = 0; i < original.cols; i++)
	{
		for (int j = 0; j < original.rows; j++)
		{
			vec3 E = vec3(0,0,1);
			double norm[3];
			double real_d = d - (double)depth.at<Vec3b>(Point(i, j))[0];
			for (int p = 0; p < 3; p++)
			{
				norm[p] = (double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2.0 * norm[p] / 255.0 - 1);
			}
			if (zAdjust)
				norm[0] = (double)zAdjust/100.0;
			
			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double color[3] = { 0,0,0 };
			for (int m = 0; m < steps; m++)
				for (int n = 0; n < steps; n++)
				{
					if (glossy)
						r_exp = dis(gen);
					vec3 Re = -1.0f *E + 2.0f*dot(N, E)*N;

					if (glossy)
					{

						double s1 = dis(gen);
						double s2 = dis(gen);
						double s3 = dis(gen);
						Re = Re + vec3(s1, s2, s3);
					}

					//printf("%f,%f,%f\n", Re[0], Re[1], Re[2]);
					float scale = (float)(real_d / fabs((double)Re[2] + 0.0001));

					int x = i + Re[0] * scale + env_width / 2 - original.cols / 2;
					int y = j + Re[1] * scale + env_height / 2 - original.rows / 2;
					x = x % env_width;
					y = y % env_height;
					if (x < 0)
						x = -x;

					if (y < 0)
						y = -y;

					
					for (int p = 0; p < 3; p++)
					{
						color[p] = color[p] + (double)env.at<Vec3b>(Point(x, y))[p];
					}
				}
			double ks = 0.5;
			for (int p = 0; p < 3; p++)
			{
				ret.at<Vec3b>(Point(i, j))[p] = (1 - ks)*original.at<Vec3b>(Point(i, j))[p] + ks * color[p]/(steps*steps);
			}
			
		}
	}
	return ret;
}

Mat diffuse(vec3 LS, Mat & dark,Mat &light, Mat & normal, Mat &depth, int zAdjust =0)
{
	Mat ret = dark.clone();
	for (int i = 0; i < dark.cols; i++)
	{ 
		for (int j = 0; j < dark.rows; j++)
		{
			vec3 L = normalize(vec3(LS[0] - i, LS[1] - j, LS[2] - depth.at<Vec3b>(Point(i, j))[1]));
			double norm[3];
			for (int p = 0; p < 3; p++)
			{
				norm[p]=(double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2*norm[p] / 255.0 -1 );				
			}
			if(zAdjust)
				norm[0] = (double)zAdjust / 100.0;

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double t = ((double)dot(L, N) + 1) / 2.0;
			for (int p = 0; p < 3; p++)
			{
				double cVal = light.at<Vec3b>(Point(i, j))[p] * t*t + dark.at<Vec3b>(Point(i, j))[p] *(1-t*t);
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
		}
	}	
	return ret;
}

Mat specular(vec3 LS, Mat diffused, Mat normal, Mat specular, Mat depth,int zAdjust = 0)
{
	Mat ret = diffused.clone();
	double s_exp = 10;

	for (int i = 0; i < diffused.cols; i++)
	{
		for (int j = 0; j < diffused.rows; j++)
		{
			vec3 L = normalize(vec3(LS[0] - i, LS[1] - j, LS[2] - depth.at<Vec3b>(Point(i, j))[1]));
			double norm[3];
			for (int p = 0; p < 3; p++)
			{
				norm[p] = (double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2 * norm[p] / 255.0 - 1);
			}
			if (zAdjust)
				norm[0] = (double)zAdjust / 100.0;

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));

			vec3 R = -1.0f * L + 2.0f *(dot(L, N)*N);
			vec3 E = vec3(0, 0, 1);
			double s = ((double)dot(R, E) + 1) / 2.0;
			//double ks = 0.9;
			for (int p = 0; p < 3; p++)
			{
				double cVal = ret.at<Vec3b>(Point(i, j))[p] * (1 - pow(s,s_exp)) + specular.at<Vec3b>(Point(i, j))[p] * (pow(s, s_exp));
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
		}
	}
	return ret;
}

double emboss(Mat original, int direction, double x, double y,int steps,double threshold =20)    //direction 0: horizontal, 1:vertical
{
	double result = 0;

	mat3 hori = mat3{
		-1,0,1,
		-2,0,2,
		-1,0,1
	};
	mat3 vert = mat3{
	-1,-2,-1,
	0,0,0,
	1,2,1
	};

	mat3x3 ori = mat3{ 0.0f };
	for(int i=0;i<3;i++)
		for (int j =0; j < 3; j++)
		{
			int x0 = x - (1 + i)*steps;
			int y0 = y - (1 + j)*steps;
			if (x0 < 0)
				x0 = 0;
			if (y0 < 0)
				y0 = 0;
			if (x0 >=original.cols)
				x0 = original.cols - 1;
			if (y0 >=original.rows)
				y0 = original.rows - 1;
			ori[i][j] = original.at<Vec3b>(Point(x0, y0))[0];
		}
	mat3 reM;
	if (direction == 0)
	{
		reM = matrixCompMult(hori, ori);
	}
	if (direction == 1)
	{
		reM = matrixCompMult(vert, ori);
	}

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			result += reM[i][j];
	if( fabs(result )< threshold)
		result = 0;
	return result/4.0;
}

void find_point(Mat &mat, int x, int y, int &a, int &b)
{
	a = x;
	b = y;
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	if (a >= mat.cols)
		a = 2 * (mat.cols - 1) - a;
	if (b >= mat.rows)
		b = 2 * (mat.rows - 1) - b;
}

Mat depth2Normal(Mat mat,double para)  //the size should be odd
{
	Mat ret = mat.clone();
	for (int i = 0; i < ret.cols; i++)
	{
		for (int j = 0; j < ret.rows; j++)
		{
			double vec[3];

			vec[1] = emboss(mat,0,i,j,para,20);

			vec[2] = emboss(mat, 1, i, j, para,20);
			//vec[1] = vec[1] / 10.0 ;
			//vec[2] = vec[2] / 10.0;
			vec[0] = 1;
			//printf("%lf\n",vec[1]);
			vec3 N = normalize(vec3(vec[0], vec[1], vec[2]));
			for (int channels = 0; channels < 3; channels++)
			{
				double color = (N[channels] + 1.0f) * 255.0f /2.0f;
				ret.at<Vec3b>(Point(i, j))[channels] = color;
			}
		}
	}
	return ret;
}



Mat shpereGradientField(Mat &mat, int x0,int y0,double r)
{
	Mat ret = mat.clone();
	for (int i = 0; i < ret.cols; i++)
	{
		for (int j = 0; j < ret.rows; j++)
		{
			double vec[3];

			vec[1] = (inCircle(i, j - 1, x0, y0, r) - inCircle(i, j + 1, x0, y0, r))*r / 2.0; 

			vec[2] = (inCircle(i - 1, j, x0, y0, r) - inCircle(i + 1, j, x0, y0, r))*r / 2.0;
			//vec[1] = vec[1] / 10.0 ;
			//vec[2] = vec[2] / 10.0;
			vec[0] = inCircle(i, j , x0, y0, r);
			if (!vec[0])
				vec[0] = 1;
			//printf("%lf\n",vec[1]);
			vec3 N = normalize(vec3(vec[0], vec[1], vec[2]));
			for (int channels = 0; channels < 3; channels++)
			{
				double color = (N[channels] + 1.0f) * 255.0f / 2.0f;
				ret.at<Vec3b>(Point(i, j))[channels] = color;
			}
		}
	}
	return ret;
}


Mat depthDiffuse(vec3 LS, Mat & dark, Mat &light, Mat & normal, Mat &depth, bool zAdjust = FALSE)
{
	Mat ret = dark.clone();
	for (int i = 0; i < dark.cols; i++)
	{
		for (int j = 0; j < dark.rows; j++)
		{
			vec3 L = normalize(vec3(LS[0] - i, LS[1] - j, LS[2] - depth.at<Vec3b>(Point(i, j))[1]));
			double norm[3];
			for (int p = 0; p < 3; p++)
			{
				norm[p] = (double)normal.at<Vec3b>(Point(i, j))[p];
				norm[p] = (2 * norm[p] / 255.0 - 1);
			}
			if (zAdjust)
				norm[0] = (double)zAdjust / 100.0;

			vec3 N = normalize(vec3(norm[2], norm[1], norm[0]));
			double t = ((double)dot(L, N) + 1) / 2.0;
			for (int p = 0; p < 3; p++)
			{
				double cVal = light.at<Vec3b>(Point(i, j))[p] * t + dark.at<Vec3b>(Point(i, j))[p] * (1 - t);
				ret.at<Vec3b>(Point(i, j))[p] = cVal;
			}
		}
	}
	return ret;
}

void circleGeneration(int x0, int y0 , int r, Mat & original, Mat & depth, Mat &normal,Mat &specular)
{

	for (int i = 0; i < depth.cols; i++)
		for (int j = 0; j < depth.rows; j++)
		{
			double z = 0;
			if ((z = jitteringCircle(i, j, x0, y0, r)) >0.1)
			{
				Vec3b color = depth.at<Vec3b>(Point(i, j));
				z *= r;
				color[0] = z/ r * 255;
				color[1] = z/ r * 255;
				color[2] = z/ r * 255;
				depth.at<Vec3b>(Point(i, j)) = color;
				original.at<Vec3b>(Point(i, j))[0] = 255;
				original.at<Vec3b>(Point(i, j))[1] = 255;
				original.at<Vec3b>(Point(i, j))[2] = 255;
				vec3 colorVec = normalize(vec3(z, (j - y0) , (i - x0)));
				normal.at<Vec3b>(Point(i, j))[0] = (colorVec[0] + 1) / 2.0 * 255;
				normal.at<Vec3b>(Point(i, j))[1] = (colorVec[1] + 1)/ 2.0 * 255;
				normal.at<Vec3b>(Point(i, j))[2] = (colorVec[2] + 1) / 2.0 * 255;
				//specular.at<Vec3b>(Point(i, j))[0] = 255;
				//specular.at<Vec3b>(Point(i, j))[1] = 255;
				//specular.at<Vec3b>(Point(i, j))[2] = 255;
			}
		}
}

int main()
{
	Mat bg = cv::imread("background.jpg", IMREAD_COLOR);
	Mat env = cv::imread("wallpaper.jpg", IMREAD_COLOR);


	//Normal map result
	//Mat original = imread("original.jpg", IMREAD_COLOR);
	//Mat normal = imread("SM.png", IMREAD_COLOR);
	//Mat dark = imread("DI0.jpg", IMREAD_COLOR);
	//Mat light = imread("DI1.jpg", IMREAD_COLOR);
	//imshow("normal", normal);
	//Mat depth(original.rows, original.cols, CV_8UC3, Scalar(0, 0, 0));
	//imshow("depth", depth);
	//vec3 lightSource = vec3(100, 200, 400);
	//Mat specularMap(original.rows, original.cols, CV_8UC3, Scalar(255, 255, 255 ));
	//Mat diffused = diffuse(lightSource, dark,light, normal, depth,TRUE);

	//Mat refr = refraction(-300, diffused, normal, bg, depth, 2.0/3.0, 8, 120);
	//Mat refl = reflection(300, diffused, normal, env, depth, 0, 170);
	//Mat f = fresnel(normal, refl, refr,100);
	//imwrite("normalresult.jpg", f);
	//imwrite("normalrefr.jpg", refr);
	//***********************
	//depth map result
	//vec3 lightSource = vec3(100, 200, 400);
	//Mat original = imread("horse.jpg", IMREAD_COLOR);
	//Mat depth = imread("horse.jpg", IMREAD_COLOR);
	////imshow("normal", normal);
	//Mat emboss = depth2Normal(depth, 5);
	//imshow("emboss", emboss);
	//imwrite("emboss.jpg", emboss);
	//Mat light = colorLift(original, 100, 0, 3);
	//Mat dark = colorLift(original, -255, 0, 3);
	//Mat diffused = diffuse(lightSource, dark,light, emboss, depth,TRUE);
	//Mat specularMap(original.rows, original.cols, CV_8UC3, Scalar(255, 255, 255 ));
	//Mat speculared = specular(lightSource, diffused, emboss, specularMap, depth, TRUE);
	//Mat refr = refraction(-300, diffused, emboss, bg, depth, 2.0/3.0, 8, 0);
	//Mat refl = reflection(300, diffused, emboss, env, depth, 0, 150);
	//Mat f = fresnel(emboss, refl, refr);
	//imwrite("depthresult.jpg", f);
	//waitKey(0);
	/*********************/
	Mat original(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	Mat depth(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	Mat normal(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255 / 2.0, 255 / 2.0));
	Mat specularMap(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255, 255));
	Mat flattenDepth(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	circleGeneration(400,400,300,original, depth, normal, specularMap);

	//cv::imshow("normal", normal);
	Mat light = colorLift(original, 0, 0, 3);
	Mat dark = colorLift(original, -255, 0, 3);
	vec3 lightSource = vec3(0, 0, 100);
	Mat diffused = diffuse(lightSource, dark,light, normal, depth);
	//cv::imshow("dif", diffused);
	/*Mat speculared = specular(lightSource, diffused, normal, specularMap, depth);
	cv::imshow("speculared", speculared);*/
	/*Mat ref = reflection(300, diffused, normal, env, depth,0,120);*/
	Mat refr = refraction(-300, diffused, normal, bg, depth, 2.0/3.0, 0, 120);
	Mat refl = reflection(300, diffused, normal, env, depth, 0, 150);
	
	//imshow("dif", diffused);
	//imshow("ref", speculared);
	cv::imshow("refr", refr);
	cv::imshow("refl", refl);
	
	Mat f = fresnel(normal, refl, refr);
	Mat speculared = specular(lightSource, f, normal, specularMap, depth);
	cv::imshow("speculared", speculared); 
	cv::imwrite("spherefakeFresnel.png", speculared);
	cv::imshow("f", f);
	cv::waitKey(0);

	/*********************/
	//Mat original(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	//Mat depth(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	//Mat normal(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255 / 2.0, 255 / 2.0));
	//Mat specularMap(WIDTH, HEIGHT, CV_8UC3, Scalar(255, 255, 255));
	//Mat flattenDepth(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	//circleGeneration(400,400,300,original, depth, normal, specularMap);

	////cv::imshow("normal", normal);
	//Mat light = colorLift(original, 0, 0, 3);
	//Mat dark = colorLift(original, -255, 0, 3);
	//vec3 lightSource = vec3(0, 0, 100);

	//normal = shpereGradientField(depth, 400,400,300);
	//imshow("normal", normal);

	//Mat diffused = diffuse(lightSource, dark,light, normal, depth);
	////cv::imshow("dif", diffused);
	///*Mat speculared = specular(lightSource, diffused, normal, specularMap, depth);
	//cv::imshow("speculared", speculared);*/
	///*Mat ref = reflection(300, diffused, normal, env, depth,0,120);*/
	//Mat refr = refraction(-300, diffused, normal, bg, depth, 2.0/3.0, 4, 120);
	//Mat refl = reflection(300, diffused, normal, env, depth, 0, 150);

	////imshow("ref", speculared);
	//cv::imshow("refr", refr);
	//cv::imshow("refl", refl);
	//
	//Mat f = fresnel(normal, refl, refr);
	//Mat speculared = specular(lightSource, f, normal, specularMap, depth);
	//cv::imshow("speculared", speculared); 
	//cv::imwrite("spherefakeFresnel.png", speculared);
	//cv::imshow("f", f);
	//cv::waitKey(0);
}

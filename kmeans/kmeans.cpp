#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <string>
#include <cstring>
#include <iostream>

using namespace cv;
using namespace std;

float* random_points(int n, int k, int dense, long s)
{
    int base = 100, step = 100, i, addon;
    float *data = (float*)malloc(sizeof(float) * n * 2);
    if (dense) {
        step = 50;
    }

    if (s == 0) {
        srand(time(NULL));
    } else {
        srand(s);
    }
    for (i = 0 ; i < n ; i++) {
        data[i * 2] = (rand() % base) + (i % k) * step;
        data[i * 2 + 1] = (rand() % base) + (i % k) * step;
    }

    return data;
}

Scalar** random_color(int n)
{
    srand(1000);
    Scalar **result = (Scalar**)malloc(sizeof(*result) * n);
    for (int i = 0 ; i < n ; i++) {
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        result[i] = new Scalar(r, g, b);
    }
    return result;
}

void draw_points(float *points, Mat label, Mat center, int n, int k)
{
    Scalar **palette = random_color(k);
    int width = 100 * k, height = width;
    Mat img(width, height, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0 ; i < n ; i++) {
        int x = (int)points[i * 2];
        int y = (int)points[i * 2 + 1];
        int c = label.at<int>(i, 0);
        circle(img, Point(x, y), 1, *palette[c]);
    }

    for (int i = 0 ; i < k ; i++) {
        int x = (int)center.at<float>(i, 0);
        int y = (int)center.at<float>(i, 1);
        drawMarker(img, Point(x, y), *palette[i], MARKER_TILTED_CROSS, 10, 1);
    }

    imshow("kmeans", img);
    while(true) {
        int key = waitKey(100);
        if (key == 'q') {
            break;
        }
    }

    // clean
    for (int i = 0 ; i < k ; i++) {
        delete palette[i];
    }
    free(palette);
}

float elbow(float *points, Mat label, Mat centers, int n, int k)
{
    float sse = 0;
    for (int i = 0 ; i < n ; i++) {
        float x = points[i * 2];
        float y = points[i * 2 + 1];
        int c = label.at<int>(i, 0);

        float cx = centers.at<float>(c, 0);
        float cy = centers.at<float>(c, 1);

        float dx = x - cx, dy = y - cy;
        sse += dx * dx + dy * dy;
    }

    sse = sqrt(sse);

    return sse;
}

int main(int argc, char **argv)
{
    long s = 0;
    int n = 200;
    int k = 3;
    int l = 3;
    int d = 0;
    int e = 0;
    int r = 10;

    char c;
    while ((c = getopt(argc, argv, "hedn:k:l:s:")) != -1) {
        switch (c) {
            case 'n':
                n = atoi(optarg);
                break;

            case 'k':
                k = atoi(optarg);
                break;

            case 'l':
                l = atoi(optarg);
                break;

            case 'd':
                d = 1;
                break;

            case 'e':
                e = 1;
                break;

            case 's':
                s = atoi(optarg);
                break;

            case 'h':
                cout << "A simple program that demonstrats opencv kmeans clustering." << endl;
                cout << "Usage: kmeans [sednklh]" << endl;
                cout << "\t-s\tset random seed manually" << endl;
                cout << "\t-e\tprint elbow statistics, without this option, " << endl;
                cout << "\t\tan image plot will be displayed" << endl;
                cout << "\t-d\tdense random point" << endl;
                cout << "\t-n\tset number of points" << endl;
                cout << "\t-k\tset true cluster number of points" << endl;
                cout << "\t-l\tset guess cluster number of points" << endl;
                cout << "\t-h\tprint this message" << endl;
                cout << "examples:" << endl;
                cout << "\tkmeans -k 5 -l 7 -n 400" << endl;
                cout << "\tkmeans -k 5 -e" << endl;
                cout << "\tkmeans -k 5 -e -s 12345" << endl;
                return 0;
        }
    }

    float *data = random_points(n, k, d, s);
    Mat m(n, 1, CV_32FC2, data);
    Mat labels, centers;
    TermCriteria termCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1);

    if (e) {
        const int LOOP = 20;
        float sse[LOOP];
        for (int i = 2 ; i < LOOP ; i++) {
            kmeans(m, i, labels, termCriteria, r, KMEANS_PP_CENTERS, centers);
            sse[i] = elbow(data, labels, centers, n, i);
        }

        cout << "------------------------------------------" << endl;
        cout << "k\tsse\t\td(sse)" << endl;
        cout << "------------------------------------------" << endl;
        for (int i = 2 ; i < LOOP ; i++) {
            if (i == 2) {
                cout << i << "\t" << sse[i] << endl;
            } else {
                float v = (sse[i - 1] - sse[i]) / sse[i];
                cout << i << "\t" << sse[i] << "\t\t" << v << endl;
            }
        }
        cout << "------------------------------------------" << endl;
    } else {
        kmeans(m, l, labels, termCriteria, r, KMEANS_PP_CENTERS, centers);

        draw_points(data, labels, centers, n, max(k, l));
    }

    // clean
    free(data);

    return 0;
}

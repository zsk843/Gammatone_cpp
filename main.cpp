#include <iostream>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<complex>
#include<cmath>
#include "fft.c"

using namespace std;
using namespace Eigen;

float** GammaToneFilters(int nfft, int sample_rate, int num_features, double width, int min_frequency, int max_frequency) {


    int output_size = nfft / 2 + 1;
    double EarQ = 9.26449;
    double pi =  3.141592653589793;
    double minBW = 24.7;
    double order = 1;
    double GTord = 4;
    Eigen::ArrayXd channel_indices;
    channel_indices = Eigen::ArrayXd(num_features);


    double res_double[output_size];
    for(int j = 0;j < num_features; j++){
        channel_indices(j) = (double)(64-j);
    }


    ArrayXd cfreqs = (-EarQ) * minBW + (channel_indices * (log(min_frequency + EarQ * minBW)-log(max_frequency + EarQ * minBW)) / num_features).exp() * (max_frequency + EarQ * minBW);

    Eigen::ArrayXd ucirc1 = Eigen::ArrayXd::LinSpaced(output_size,0,output_size-1);

    complex<double> j = complex<double>(0.0,1.0);
    j = j * pi * 2.0;
    ArrayXcd tmp1 = ucirc1 * j;

    ArrayXcd ucirc = (tmp1 / nfft).exp();

    float** res  = new float*[output_size];
    for(int i = 0;i < output_size;i++)
        res[i] = new float[num_features];

    for(int i = 0; i < num_features; i++){

        double cf = cfreqs(i);
        double ERB = (double)width * (pow((cf / EarQ),order) + pow(pow(minBW,order),(1.0 / order)));

        double B = 1.019 * 2.0 * pi * ERB;
        complex <double> exp_tmp = (-B) / (double)sample_rate;
        complex<double> r = exp(exp_tmp);
        double theta = 2.0 * pi * cf / (double)sample_rate;
        j = complex<double>(0.0,1.0);
        complex<double> tmp_com1 = j*theta;
        complex<double> pole = r* exp(tmp_com1);

        double T = 1.0f / sample_rate;
        double A11 = -(2.0 * T * cos(2.0 * cf * pi * T) / exp(B * T) + 2.0 * sqrt(3.0 + pow(2.0 ,1.5)) * T * sin(2.0 * cf * pi * T) / exp(B * T)) / 2.0;
        double A12 = -(2.0 * T * cos(2.0 * cf * pi * T) / exp(B * T) - 2.0 * sqrt(3.0 + pow(2.0 ,1.5)) * T * sin(2.0 * cf * pi * T) / exp(B * T)) / 2.0;
        double A13 = -(2.0 * T * cos(2.0 * cf * pi * T) / exp(B * T) + 2.0 * sqrt(3.0 - pow(2.0 ,1.5)) * T * sin(2.0 * cf * pi * T) / exp(B * T)) / 2.0;
        double A14 = -(2.0 * T * cos(2.0 * cf * pi * T) / exp(B * T) - 2.0 * sqrt(3.0 - pow(2.0 ,1.5)) * T * sin(2.0 * cf * pi * T) / exp(B * T)) / 2.0;


        j = complex<double>(0.0,1.0);
        complex<double> T1 = exp(4.0 *    cf * pi * T * j  );
        complex<double>T2 = exp(-(B * T) + 2.0 * j * cf * pi * T);
        double T3 = cos(2.0 * cf * pi * T);
        double T41 = sqrt(3.0 + pow(2.0, 1.5)) * sin(2.0 * cf * pi * T);
        double T42 = sqrt(3.0 - pow(2.0, 1.5)) * sin(2.0 * cf * pi * T);

        complex<double>T51 = -2.0 * T1 * T + 2.0 * T2 * T * (T3 + T41);
        complex<double>T52 = -2.0 * T1 * T + 2.0 * T2 * T * (T3 - T41);
        complex<double>T53 = -2.0 * T1 * T + 2.0 * T2 * T * (T3 + T42);
        complex<double>T54 = -2.0 * T1 * T + 2.0 * T2 * T * (T3 - T42);
        complex<double>T6 = -2.0 / exp(2.0 * B * T) - 2.0 * T1 + 2.0 * (1.0 + T1) / exp(B * T);

        complex<double> gain = abs((T51 * T52 * T53 * T54) / pow(T6 , 4.0));
        complex<double> conjugate_pole = complex<double>(pole.real(), -pole.imag());
        Eigen::ArrayXcd tmp_array1 = (pow(T , 4.0) / gain)*(ucirc + complex<double>((A11 / T),0)).abs()*(ucirc + A12 / T).abs()*(ucirc + A13 / T).abs()*(ucirc + A14 / T).abs();

        Eigen::ArrayXcd tmp_array2 = ((-ucirc+pole)*(-ucirc+conjugate_pole)).abs();

        Eigen::ArrayXcd res1 = tmp_array1*(tmp_array2.pow(-GTord));


        for(int k = 0;k < output_size; k++) {
            res[k][i] = (float)res1(k).real();
        }
    }
    return res;
}

float** GammaToneFeature(short* data, int data_len, float* factors, int* ifac, int frame_length, int frame_interval, float** filters){

    float g_floor = (float)exp(-50);
    float pre_emp = 0.97;
    int nfft = (int)pow(2, (int)(ceil(log(frame_length) / log(2))));

    int samples_len = data_len;
    float* samples = new float[samples_len];
    samples[0] = data[0]/32768.0f;
    for(int i = 0; i < samples_len-1;i++)
        samples[i+1] = data[i+1] - pre_emp* data[i];

    int hamming_len = (samples_len+1)/2;
    float win[hamming_len];
    for( int i=0; i<hamming_len; ++i )
    {
        win[i] = 1.0f * (float)(0.54 - 0.46*cos((3.141592653589793)*i/(frame_length-1.0)));
        win[frame_length-1-i] = win[i];
    }

    int frame_num = 1+(int)(floor((samples_len-frame_length)/frame_interval));
    int start_index;
    float tmp_data[nfft];
    float zeros[nfft-frame_length];
    memset(zeros,0,sizeof(float)*(nfft-frame_length));
    float** res = new float*[frame_num];

    for(int i = 0; i < frame_num; i++)
    {
        start_index = i*frame_length;
        memcpy(tmp_data,samples+start_index, sizeof(float)*frame_length);
        memcpy(tmp_data+frame_length,zeros,sizeof(float)*(nfft-frame_length));
        drftf1(nfft, tmp_data, factors,factors+nfft,ifac);
        res[i] = tmp_data;
    }

    return res;
}

float* FFTFactors(int n){
    int output_dim = 4*n + 15;
    int ifc_dim = n+15;

    float factors[output_dim];
    int ifa[ifc_dim];

    drfti1(n,factors+n,ifa);

    float ifac[ifc_dim+output_dim];
    for(int i = 0; i < output_dim; i++)
        ifac[i] = factors[i];
    for(int i = 0; i <ifc_dim; i++)
        ifac[i+output_dim] = ifa[i];

    return ifac;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    float** res = GammaToneFilters(512,16000,64,0.5,50,8000);
    return 0;
}


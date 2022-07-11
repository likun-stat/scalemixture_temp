/* p_integrand.c */
#include <math.h>
double f(int n, double *x, void *user_data) {
    double delta = *(double *)user_data;
    double x_inte = x[0];
    
    double xval = x[1];
    double tau_sqd = x[2];
    double tmp1 = x[3];
    double tmp2 = x[4];
    double tmp3 = x[5];
    double tmp4 = x[6];
    double half_result = tmp1*pow(xval-x_inte,tmp2)-tmp3/(xval-x_inte);
    double dnorm = exp(-x_inte*x_inte*tmp4);
     
    return -dnorm+dnorm*half_result;
}

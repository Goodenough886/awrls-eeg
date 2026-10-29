// Execution-only VME iteration kernel. Same equations and stopping rule as
// blink_repro.methods.ovme_reg.VMEWorkspace.extract (Nazari BSD port).
// No fast-math, unchanged HHO RNG, objective, parameters and FFT reconstruction.
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
static double pairwise(const double* a,int n){
    if(n<8){double r=-0.0;for(int i=0;i<n;i++)r+=a[i];return r;}
    if(n<=128){double r[8];for(int j=0;j<8;j++)r[j]=a[j];int i=8;
        for(;i<n-(n%8);i+=8)for(int j=0;j<8;j++)r[j]+=a[i+j];
        double z=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(;i<n;i++)z+=a[i];return z;}
    int n2=n/2;n2-=n2%8;return pairwise(a,n2)+pairwise(a+n2,n-n2);
}
extern "C" __declspec(dllexport) int vme_iterate(int n,const double* freq,const double* spec,
 double alpha,double omega,double tau,double tol,int maxit,double* out,double* history,double* meta){
    std::vector<double> re(n,0),im(n,0),dr(n,0),di(n,0),nr(n),ni(n),power(n);
    history[0]=omega;double change=INFINITY;int step=0;
    for(step=1;step<maxit;step++){
        for(int k=0;k<n;k++){
            double delta=freq[k]-omega;double d2=delta*delta;double ker=(alpha*alpha)*(d2*d2);
            double den=(1+ker)*(1+2*ker);
            nr[k]=(spec[2*k]+re[k]*ker+dr[k]/2)/den;
            ni[k]=(spec[2*k+1]+im[k]*ker+di[k]/2)/den;
            double ab=std::hypot(nr[k],ni[k]);power[k]=ab*ab;
        }
        double total=pairwise(power.data(),n),dot=0;
        for(int k=0;k<n;k++)dot+=freq[k]*power[k];
        if(total>0)omega=dot/total;
        double cr=0,ci=0;
        for(int k=0;k<n;k++){
            if(tau!=0){double diff=freq[k]-omega;double ker=(alpha*alpha)*std::pow(diff,4);
                double rr=ker*(spec[2*k]-nr[k])/(1+2*ker),ri=ker*(spec[2*k+1]-ni[k])/(1+2*ker);
                dr[k]+=tau*(spec[2*k]-nr[k]-rr);di[k]+=tau*(spec[2*k+1]-ni[k]-ri);}
            double xr=nr[k]-re[k],xi=ni[k]-im[k];cr+=xr*xr-xi*xi;ci+=2*xr*xi;
            re[k]=nr[k];im[k]=ni[k];
        }
        change=std::hypot(std::numeric_limits<double>::epsilon()+cr/(2*n),ci/(2*n));
        history[step]=omega;if(change<=tol)break;
    }
    int used=std::min(step,maxit-1);
    for(int k=0;k<n;k++){out[2*k]=re[k];out[2*k+1]=im[k];}
    meta[0]=change;meta[1]=omega;return used;
}

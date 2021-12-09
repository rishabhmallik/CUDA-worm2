//cuda worm 2 code
#include <iostream>
#include <math.h>
#include <random>
#include <time.h>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include "parameters.h"

long long int time_a,time_k;

using namespace std::chrono;
using namespace std;

double Rand() {
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(0, 1);
    return dis(gen);
}
__host__ __device__ int mod(int a,int b){
    return (a%b + b)%b;
}

__device__ __host__ void shiftx(int b[], int a[], int sign, int v){
    for (int i=0; i<d; i++){
        if (i==v){
            if (v==0){ b[i]=mod(a[i]+sign,Nt);  }
            else { b[i]=mod(a[i]+sign,Ns);  }
        }
        else { b[i]=a[i]; }
    }
}

double I(int s){
    double a=0,r=0;
    while(r<inf){
        a+=dr*pow(r,s+1)*exp(-eta*pow(r,2)-lmd*pow(r,4));
        r+=dr;
    }
    return a;
}

__host__ __device__ int sx(int x[], int k[], int a[]){
    int sum=0;
    int v[d]={0};
    for (int i=0;i<d;i++){
        v[i]=1;
        
        sum+=abs(k[x[0]+Nt*x[1]+Nt*Ns*i])
        +abs(k[mod(x[0]-v[0],Nt)+Nt*mod(x[1]-v[1],Ns)+Nt*Ns*i])
        +2*(a[x[0]+Nt*x[1]+Nt*Ns*i]
        +a[mod(x[0]-v[0],Nt)+Nt*mod(x[1]-v[1],Ns)+Nt*Ns*i]);

        v[i]=0;
    }
    return sum;
}

__device__ bool eql(int x1[],int x2[]){
    bool flag=true;
    for (int i=0;i<d;i++){
        if(x1[i]!=x2[i]){
            flag=false;
            break;
        }
    }
    return flag;
}

__global__ void a_update(int t, int tag, int *k, int *a, int *a_, double *I_val){
    int y, x[d], x_[d];
    double rho;
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    
    if (id>Ns*Nt/2-1) return;
    if (t==0){
        x[0]=(2*id+tag)%Nt;
        x[1]=(2*id+tag)/Nt;
    }
    else {
        x[0]=(2*id+tag)/Ns;
        x[1]=(2*id+tag)%Ns;
    }
    
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    double rand1 = curand_uniform_double(&state);
    int rand2 = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    y=a_[x[0]+Nt*x[1]+Nt*Ns*t];
    a_[x[0]+Nt*x[1]+Nt*Ns*t]=a[x[0]+Nt*x[1]+Nt*Ns*t]+rand2;
    
    if (a_[x[0]+Nt*x[1]+Nt*Ns*t]<0){
        a_[x[0]+Nt*x[1]+Nt*Ns*t]=y;
        return;
    }
    
    shiftx(x_, x, 1, t);
    if (a_[x[0]+Nt*x[1]+Nt*Ns*t]>a[x[0]+Nt*x[1]+Nt*Ns*t]){
        rho=1.0/(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a_[x[0]+Nt*x[1]+Nt*Ns*t])
        /a_[x[0]+Nt*x[1]+Nt*Ns*t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    } 
    else{
        rho=1.0*(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t])
        *a[x[0]+Nt*x[1]+Nt*Ns*t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    }
    if (rand1<rho){
        //printf("ar\n");
        a[x[0]+Nt*x[1]+Nt*Ns*t]=a_[x[0]+Nt*x[1]+Nt*Ns*t];
    }
    else{
        a_[x[0]+Nt*x[1]+Nt*Ns*t]=y;
    }
}
__device__ int countl(int *l){
    int sum=0;
    for (int i=0; i<Nt; i++){
        for (int j=0; j<Ns; j++){
            sum+=l[i+j*Nt];
        }
    }
    return sum;
}
__host__ __device__ int flux(int *k){
    int sum=0,sump;
    int v[d]={0}, x[d];
    for (int p=0;p<Nt;p++){
        for (int q=0;q<Ns;q++){
            sump=0;
            x[0]=p; x[1]=q;
            for (int i=0;i<d;i++){
                v[i]=1;
                
                sump+=k[x[0]+Nt*x[1]+Nt*Ns*i]
                -k[mod(x[0]-v[0],Nt)+Nt*mod(x[1]-v[1],Ns)+Nt*Ns*i];
                
                v[i]=0;
            }
            sum+=abs(sump);
        }
    }
    
    return sum;
}

__global__ void worm_update(int *k, int *k_, int *a, double mu, double *I_val, int *l, int *sp, int *sm, int worms){
    
    double rand1, rho=1.0;
    int sign, v, y, del;
    
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>=worms) return;
    
    curandState state;
    
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    
    int x[d], x0[d], xx[d], x_[d], flag;
    
	bool start=false;
	while (!start){
		
		for (int i=1; i<d; i++){
			x0[i]=(int)(Ns*(1-curand_uniform_double(&state)));
		}
		x0[0]=(int)(Nt*(1-curand_uniform_double(&state)));
		
		flag=atomicExch(&l[x0[0]+Nt*x0[1]],1);
		if(flag==1) continue;
		
		//del=1;
		del=2*((int)(2*(1-curand_uniform_double(&state))))-1;
		v=(int)(d*(1-curand_uniform_double(&state)));
		sign=2*((int)(2*(1-curand_uniform_double(&state))))-1;
		
		shiftx(x,x0,sign,v);
		
		flag=atomicExch(&l[x[0]+Nt*x[1]],1);
		if(flag==1) {
			l[x0[0]+Nt*x0[1]]=0;
			continue;
		}
		
		rand1=curand_uniform_double(&state);
		
		if (sign<0){
			shiftx(xx,x0,-1,v);
		}
		else{
			shiftx(xx,x0,0,v);
		}
		int xx_pos=xx[0]+Nt*xx[1]+Nt*Ns*v;
		y=k_[xx_pos];
		k_[xx_pos]=k[xx_pos]+del*sign;
		if(abs(k_[xx_pos])>abs(k[xx_pos])){
			rho=exp(sign*mu*del*(v==0))
			/(abs(k_[xx_pos])+a[xx_pos])
			*A/I_val[sx(x,k,a)]/I_val[sx(x0,k,a)];
		}
		else {
			rho=exp(sign*mu*del*(v==0))
			*(abs(k[xx_pos])+a[xx_pos])
			*A/I_val[sx(x,k,a)]/I_val[sx(x0,k,a)];
		}
		
		if (rand1<rho){
			k[xx_pos]=k_[xx_pos];
			if (del>0) sp[x0[0]+Nt*x0[1]]+=1;
			else sm[x0[0]+Nt*x0[1]]+=1;
			l[x0[0]+Nt*x0[1]]=0;
			start=true;
		}
		else{
			k_[xx_pos]=y;
			l[x0[0]+Nt*x0[1]]=0;
			l[x[0]+Nt*x[1]]=0;
		}
		
	}
	
	//clock_t start_t= clock();
	bool flag1=true;
	
	while (flag1){
		v=(int)(d*(1-curand_uniform_double(&state)));
		sign=2*((int)(2*(1-curand_uniform_double(&state))))-1;
		shiftx(x_,x,sign,v);
		
		int x_pos=x_[0]+Nt*x_[1];
		
		flag=atomicExch(&l[x_pos],1);
		if (flag==1) continue;
		
		rand1=curand_uniform_double(&state);
		
		if (sign<0){
			shiftx(xx,x,-1,v);
		}
		else{
			shiftx(xx,x,0,v);
		}
		int xx_pos=xx[0]+xx[1]*Nt+v*Nt*Ns;
		y=k_[xx_pos];
		k_[xx_pos]=k[xx_pos]+del*sign;
		
		if(abs(k_[xx_pos])>abs(k[xx_pos])){
			if (del>0){
				if (sp[x_[0]+Nt*x_[1]]>0){
					rho=exp(sign*mu*del*(v==0))
					/(abs(k_[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]*I_val[sx(x_,k_,a)]/A;
				}
				else {
					rho=exp(sign*mu*del*(v==0))
					/(abs(k_[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]/I_val[sx(x_,k,a)];
				}
			}
			else{
				if (sm[x_[0]+Nt*x_[1]]>0){
					rho=exp(sign*mu*del*(v==0))
					/(abs(k_[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]*I_val[sx(x_,k_,a)]/A;
				}
				else {
					rho=exp(sign*mu*del*(v==0))
					/(abs(k_[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]/I_val[sx(x_,k,a)];
				}
			}
		}
		else {
			if (del>0){
				if (sp[x_[0]+Nt*x_[1]]>0){
					rho=exp(sign*mu*del*(v==0))
					*(abs(k[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]*I_val[sx(x_,k_,a)]/A;
				}
				else{
					rho=exp(sign*mu*del*(v==0))
					*(abs(k[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]/I_val[sx(x_,k,a)];
				}
			}
			else{
				if (sm[x_[0]+Nt*x_[1]]>0){
					rho=exp(sign*mu*del*(v==0))
					*(abs(k[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]*I_val[sx(x_,k_,a)]/A;
				}
				else{
					rho=exp(sign*mu*del*(v==0))
					*(abs(k[xx_pos])+a[xx_pos])
					*I_val[sx(x,k_,a)]/I_val[sx(x_,k,a)];
				}
			}
		}
		
		if (rand1<rho){
			k[xx_pos]=k_[xx_pos];
			l[x[0]+Nt*x[1]]=0;
			shiftx(x,x_,0,v);
		}
		else{
			k_[xx_pos]=y;
			l[x_pos]=0;
		}
		if (del>0) {
			if (sp[x[0]+Nt*x[1]]>0){
				flag1=false;
				sp[x[0]+Nt*x[1]]-=1;
				l[x[0]+Nt*x[1]]=0; 
			}
		}
		else {
			if (sm[x[0]+Nt*x[1]]>0){
				flag1=false;
				sm[x[0]+Nt*x[1]]-=1;
				l[x[0]+Nt*x[1]]=0;
			}
		}
	}
}

void update(int *k, int *k_, int *a, int *a_, double mu, double *I_val, int *l, int *sp, int *sm, int worms, bool flag2){
    
    blocks=Ns*Nt/2/blockSize+1;
    
    a_update<<<blocks,blockSize>>>(0,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(0,1,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,1,k,a,a_,I_val);
    
    blocks=worms/blockSize+1;
    
    worm_update<<<blocks,blockSize>>>(k,k_,a,mu,I_val,l,sp,sm,worms);
}


__global__ void init_lattice(int *Ar, int a, int n){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<n) Ar[i]=a;
}

int ksum(int k[]){
    int sum=0;
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            sum+=k[i+j*Nt];
        }
    }
    return sum;
}

double phi2(int k[], int a[], double I[int_val]){
    double sum=0;
    int x[d];
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            x[0]=i; x[1]=j;
            sum+=I[sx(x, k, a)+2]/I[sx(x, k, a)];
        }
    }
    return sum/Nt/Ns;
}
double phi4(int k[], int a[], double I[int_val]){
    double sum=0;
    int x[d];
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            x[0]=i; x[1]=j;
            sum+=I[sx(x, k, a)+4]/I[sx(x, k, a)];
        }
    
    }
    return sum/Nt/Ns;
}

double errorjack(double xi[], int configs){
    double *x_i, x_=0, stddev=0;
    x_i=(double*) malloc(configs*sizeof(*x_i));
    for (int i=0; i<configs; i++){
        x_i[i]=0;
        for (int j=0; j<configs; j++){
            x_i[i]+=(1-(i==j))*xi[j];
        }
        x_i[i]=x_i[i]/(configs-1);
        x_+=x_i[i];
    }
    x_=x_/configs;
    for (int i=0; i<configs; i++){
        stddev+=(x_i[i]-x_)*(x_i[i]-x_);
    }
    stddev=sqrt(stddev*(configs-1)/configs);
    return stddev;
}
__global__ void print(double *Ar){
    printf("%f\n",Ar[0]);
}

int main(int argc, char **argv)
{
    auto begin=high_resolution_clock::now();
    
    int *k, *a, *a_, *kh, *ah, *k_, *l, *sp, *sm, itr;
    double *I_val;
    
    int worms=atoi(argv[1]);
    
    cudaMalloc(&k, Nt*Ns*d*sizeof(*k));
    cudaMalloc(&k_, Nt*Ns*d*sizeof(*k_));
    cudaMalloc(&a, Nt*Ns*d*sizeof(*a));
    cudaMalloc(&a_, Nt*Ns*d*sizeof(*a_));
    cudaMalloc(&l, Nt*Ns*sizeof(*l));
    cudaMalloc(&I_val, int_val*sizeof(*I_val));
    cudaMalloc(&sp, Nt*Ns*sizeof(*sp));
    cudaMalloc(&sm, Nt*Ns*sizeof(*sm));
    
    kh=(int*) malloc(Nt*Ns*d*sizeof(*kh));
    ah=(int*) malloc(Nt*Ns*d*sizeof(*ah));
    
    double dmu=(mu_max-mu_min)/mu_n;
    double n_avg, phi2_avg, phi4_avg;
    double *xi,*phi2i, I_val_h[int_val];
    double mu=mu_min;
    
    xi=(double*) malloc(configs*sizeof(*xi));
    phi2i=(double*) malloc(configs*sizeof(*phi2i));
    
    for (int i=0; i<int_val; i++){
        I_val_h[i]=I(i);
    }
    
    cudaMemcpy(I_val, I_val_h, int_val*sizeof(*I_val), cudaMemcpyHostToDevice);
    ofstream data, data1, data2, data3, data4, data5;
    
    string filename0="mu_vs_n_worms="+to_string(worms)+".txt";
    string filename1="mu_vs_phi2_worms="+to_string(worms)+".txt";
    
    data.open(filename0);
    data1.open(filename1);
      
    for (int g=0; g<mu_n; g++){
        
        blocks=Nt*Ns*d/blockSize+1;
        init_lattice<<<blocks,blockSize>>>(k,0,Ns*Nt*d);
        init_lattice<<<blocks,blockSize>>>(k_,0,Ns*Nt*d);
        init_lattice<<<blocks,blockSize>>>(a,0,Ns*Nt*d);
        init_lattice<<<blocks,blockSize>>>(a_,0,Ns*Nt*d);
        blocks=Nt*Ns/blockSize+1;
        init_lattice<<<blocks,blockSize>>>(l,0,Ns*Nt);
        init_lattice<<<blocks,blockSize>>>(sp,0,Ns*Nt);
        init_lattice<<<blocks,blockSize>>>(sm,0,Ns*Nt);
        
        for (int i=0; i<equil; i++){
            update(k,k_,a,a_,mu,I_val,l,sp,sm,worms,false);
            
        }
        
        phi2_avg=0;
        n_avg=0;
        
        for (int i=0; i<configs; i++){
            for (int j=0; j<gaps; j++){
                update(k,k_,a,a_,mu,I_val,l,sp,sm,worms,true);
            }
            
            update(k,k_,a,a_,mu,I_val,l,sp,sm,worms,true);
            cudaDeviceSynchronize();
            cudaMemcpy(kh, k, Nt*Ns*d*sizeof(*k), cudaMemcpyDeviceToHost);
            cudaMemcpy(ah, a, Nt*Ns*d*sizeof(*a), cudaMemcpyDeviceToHost);
            
            xi[i]=1.0*ksum(kh)/Nt/Ns;
            phi2i[i]=phi2(kh,ah,I_val_h);
            
            n_avg+=xi[i];
            phi2_avg+=phi2i[i];
            
        }
        n_avg=n_avg/configs;
        phi2_avg=phi2_avg/configs;
        
		data<<mu<<"\t"<<n_avg<<"\t"<<errorjack(xi,configs)<<"\n";
        data1<<mu<<"\t"<<phi2_avg<<"\t"<<errorjack(phi2i,configs)<<"\n";
        
        mu+=dmu;
	}
    
    data1.close();
    data.close();
    
    free(kh);
    auto stop=high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-begin);
    cout<<duration.count()<<endl;
	
    return 0;
}

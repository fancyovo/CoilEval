// Isolated discrete RK4 forward/reverse. All geometry dependence is analytic.
// Field values/Jacobians are reduced in the segment dtype; states may be FP64.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int FC_TRACE_THREADS = 128;
template<class T> struct V3 { T x,y,z; };
template<class T> __device__ V3<T> cross3(V3<T>a,V3<T>b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
template<class T> __device__ T reduce(T value) {
    __shared__ T partial[4];
    for(int k=16;k;k>>=1)value+=__shfl_down_sync(0xffffffff,value,k);
    if((threadIdx.x&31)==0)partial[threadIdx.x>>5]=value;
    __syncthreads();
    T result=threadIdx.x<4?partial[threadIdx.x]:T(0);
    if(threadIdx.x<32)for(int k=16;k;k>>=1)result+=__shfl_down_sync(0xffffffff,result,k);
    __syncthreads();
    return result;
}

template<class T,class S> __device__ void rhs_stage(
    const T* positions,const T* elements,int ns,S R,S Z,S phi,
    T* xyz,S* jacobian,S* field_derivative,S* rhs) {
    // All block lanes see the same state. Math/reduction order for B matches
    // full_chain_field.cu; no point-by-segment interaction tape is retained.
    const S cosine=cos(phi),sine=sin(phi);
    const V3<T> point{T(R*cosine),T(R*sine),T(Z)};
    T sums[12]={};
    for(int segment=threadIdx.x;segment<ns;segment+=FC_TRACE_THREADS) {
        const V3<T> r{point.x-positions[3*segment],point.y-positions[3*segment+1],point.z-positions[3*segment+2]};
        const V3<T> w{elements[3*segment],elements[3*segment+1],elements[3*segment+2]};
        const T distance2=r.x*r.x+r.y*r.y+r.z*r.z;
        const T r2=fmax(distance2,T(1e-24));
        const T inverse=T(1)/sqrt(r2),inverse3=inverse*inverse*inverse;
        const V3<T> t=cross3(w,r);
        sums[0]+=t.x*inverse3;sums[1]+=t.y*inverse3;sums[2]+=t.z*inverse3;
        const T coords[3]={r.x,r.y,r.z};
        const V3<T> e[3]={{1,0,0},{0,1,0},{0,0,1}};
        for(int j=0;j<3;++j) {
            const V3<T> q=cross3(w,e[j]);
            // The clamp is constant below the distance floor, including its
            // zero derivative. The cross product still depends on position.
            const T radial=distance2>T(1e-24)?-T(3)*coords[j]*inverse3/r2:T(0);
            sums[3+j]+=q.x*inverse3+t.x*radial;
            sums[6+j]+=q.y*inverse3+t.y*radial;
            sums[9+j]+=q.z*inverse3+t.z*radial;
        }
    }
    __shared__ T b_and_g[12];
    for(int j=0;j<12;++j) {
        const T value=reduce(sums[j])*T(1e-7);
        if(threadIdx.x==0)b_and_g[j]=value;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        xyz[0]=point.x;xyz[1]=point.y;xyz[2]=point.z;
        const S bx=S(b_and_g[0]),by=S(b_and_g[1]),bz=S(b_and_g[2]);
        const S br=bx*cosine+by*sine,bp=-bx*sine+by*cosine;
        const bool active=fabs(bp)>S(1e-14);
        const S denominator=active?bp:(bp<S(0)?-S(1e-14):S(1e-14));
        rhs[0]=R*br/denominator;rhs[1]=R*bz/denominator;
        const S bp_gradient[3]={-sine,cosine,S(0)};
        const S br_gradient[3]={cosine,sine,S(0)};
        for(int j=0;j<3;++j) {
            field_derivative[j]=R*br_gradient[j]/denominator-
                (active?R*br*bp_gradient[j]/(denominator*denominator):S(0));
            field_derivative[3+j]=(j==2?R/denominator:S(0))-
                (active?R*bz*bp_gradient[j]/(denominator*denominator):S(0));
        }
        for(int component=0;component<2;++component) {
            S dR=(component==0?br:bz)/denominator,dZ=0;
            for(int k=0;k<3;++k) {
                const S m=field_derivative[3*component+k];
                dR+=m*(S(b_and_g[3+3*k])*cosine+S(b_and_g[4+3*k])*sine);
                dZ+=m*S(b_and_g[5+3*k]);
            }
            jacobian[2*component]=dR;jacobian[2*component+1]=dZ;
        }
    }
    __syncthreads();
}

template<class T,class S> __global__ void trace_forward_kernel(
    const T* positions,const T* elements,const S* starts,const S* phi0,const S* phi1,
    S* states,T* stage_xyz,S* stage_jacobian,S* stage_field_derivative,
    int ns,int lines,int steps,int queries) {
    const int global_line=blockIdx.x;if(global_line>=lines*queries)return;
    const int query=global_line/lines,line=global_line%lines;
    positions+=static_cast<size_t>(query)*ns*3;
    elements+=static_cast<size_t>(query)*ns*3;
    __shared__ S state[2],initial[2],k[8];
    const S begin=phi0[global_line],h=(phi1[global_line]-begin)/S(steps);
    if(threadIdx.x==0) {
        state[0]=starts[2*global_line];state[1]=starts[2*global_line+1];
        states[2*global_line]=state[0];states[2*global_line+1]=state[1];
    }
    __syncthreads();
    for(int step=0;step<steps;++step) {
        if(threadIdx.x==0){initial[0]=state[0];initial[1]=state[1];}
        __syncthreads();
        for(int stage=0;stage<4;++stage) {
            if(threadIdx.x==0 && stage>0) {
                const S scale=stage==3?h:h/S(2);
                state[0]=initial[0]+scale*k[2*(stage-1)];
                state[1]=initial[1]+scale*k[2*(stage-1)+1];
            }
            __syncthreads();
            const size_t at=((static_cast<size_t>(query)*steps+step)*4+stage)*lines+line;
            const S angle=begin+h*S(step)+(stage==0?S(0):(stage==3?h:h/S(2)));
            rhs_stage<T,S>(positions,elements,ns,state[0],state[1],angle,
                stage_xyz+3*at,stage_jacobian+4*at,stage_field_derivative+6*at,k+2*stage);
        }
        if(threadIdx.x==0) {
            state[0]=initial[0]+h*(k[0]+S(2)*k[2]+S(2)*k[4]+k[6])/S(6);
            state[1]=initial[1]+h*(k[1]+S(2)*k[3]+S(2)*k[5]+k[7])/S(6);
            const size_t at=(static_cast<size_t>(step)+1)*lines*queries+global_line;
            states[2*at]=state[0];states[2*at+1]=state[1];
        }
        __syncthreads();
    }
}

template<class T,class S> __global__ void trace_reverse_kernel(
    const S* jacobian,const S* field_derivative,const S* phi0,const S* phi1,
    const S* adj_states,S* adj_starts,T* adj_B,int lines,int steps,int all_states,int queries) {
    const int global_line=blockIdx.x*blockDim.x+threadIdx.x;if(global_line>=lines*queries)return;
    const int query=global_line/lines,line=global_line%lines;
    const S h=(phi1[global_line]-phi0[global_line])/S(steps);
    const size_t final_at=all_states?static_cast<size_t>(steps)*lines*queries+global_line:global_line;
    S carry[2]={adj_states[2*final_at],adj_states[2*final_at+1]};
    for(int step=steps-1;step>=0;--step) {
        S base[2]={carry[0],carry[1]},ak[8];
        for(int stage=0;stage<4;++stage) {
            const S weight=h/(stage==0||stage==3?S(6):S(3));
            ak[2*stage]=weight*carry[0];ak[2*stage+1]=weight*carry[1];
        }
        for(int stage=3;stage>=0;--stage) {
            const size_t at=((static_cast<size_t>(query)*steps+step)*4+stage)*lines+line;
            const S* J=jacobian+4*at;const S* M=field_derivative+6*at;
            const S a0=ak[2*stage],a1=ak[2*stage+1];
            for(int j=0;j<3;++j)adj_B[3*at+j]=T(M[j]*a0+M[3+j]*a1);
            const S ay[2]={J[0]*a0+J[2]*a1,J[1]*a0+J[3]*a1};
            base[0]+=ay[0];base[1]+=ay[1];
            if(stage>0) {
                const S scale=stage==3?h:h/S(2);
                ak[2*(stage-1)]+=scale*ay[0];ak[2*(stage-1)+1]+=scale*ay[1];
            }
        }
        const size_t at=static_cast<size_t>(step)*lines*queries+global_line;
        carry[0]=base[0]+(all_states?adj_states[2*at]:S(0));
        carry[1]=base[1]+(all_states?adj_states[2*at+1]:S(0));
    }
    adj_starts[2*global_line]=carry[0];adj_starts[2*global_line+1]=carry[1];
}

template<class T,class S> int forward(const void*p,const void*w,const void*y,const void*a,const void*b,
    void*states,void*xyz,void*J,void*M,int ns,int lines,int steps,std::uint64_t stream,int queries=1) {
    if(ns<=0||lines<=0||steps<=0||queries<=0||!p||!w||!y||!a||!b||!states||!xyz||!J||!M)return int(cudaErrorInvalidValue);
    trace_forward_kernel<T,S><<<lines*queries,FC_TRACE_THREADS,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const T*>(p),static_cast<const T*>(w),static_cast<const S*>(y),
        static_cast<const S*>(a),static_cast<const S*>(b),static_cast<S*>(states),
        static_cast<T*>(xyz),static_cast<S*>(J),static_cast<S*>(M),ns,lines,steps,queries);
    return int(cudaGetLastError());
}
template<class T,class S> int backward(const void*J,const void*M,const void*a,const void*b,const void*as,
    void*ay,void*ab,int lines,int steps,int all_states,std::uint64_t stream,int queries=1) {
    if(lines<=0||steps<=0||queries<=0||!J||!M||!a||!b||!as||!ay||!ab)return int(cudaErrorInvalidValue);
    trace_reverse_kernel<T,S><<<(lines*queries+127)/128,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const S*>(J),static_cast<const S*>(M),static_cast<const S*>(a),static_cast<const S*>(b),
        static_cast<const S*>(as),static_cast<S*>(ay),static_cast<T*>(ab),lines,steps,all_states,queries);
    return int(cudaGetLastError());
}
} // namespace

#define FC_TRACE_EXPORT(SUFFIX,T,S) \
extern "C" int fc_trace_forward_##SUFFIX(const void*p,const void*w,const void*y,const void*a,const void*b,void*states,void*xyz,void*J,void*M,int ns,int lines,int steps,std::uint64_t stream){return forward<T,S>(p,w,y,a,b,states,xyz,J,M,ns,lines,steps,stream);} \
extern "C" int fc_trace_reverse_##SUFFIX(const void*J,const void*M,const void*a,const void*b,const void*as,void*ay,void*ab,int lines,int steps,int all_states,std::uint64_t stream){return backward<T,S>(J,M,a,b,as,ay,ab,lines,steps,all_states,stream);}
FC_TRACE_EXPORT(f32_f32,float,float)
FC_TRACE_EXPORT(f32_f64,float,double)
FC_TRACE_EXPORT(f64_f32,double,float)
FC_TRACE_EXPORT(f64_f64,double,double)

// Batch tapes are query-major [query,step,stage,line,...], permitting one
// independent-query field VJP after the state reverse. States retain the public
// step-major shape [step,query,line,2]. Query count one is identical to above.
#define FC_BATCH_TRACE_EXPORT(SUFFIX,T,S) \
extern "C" int fc_batch_trace_forward_##SUFFIX(const void*p,const void*w,const void*y,const void*a,const void*b,void*states,void*xyz,void*J,void*M,int ns,int lines,int queries,int steps,std::uint64_t stream){return forward<T,S>(p,w,y,a,b,states,xyz,J,M,ns,lines,steps,stream,queries);} \
extern "C" int fc_batch_trace_reverse_##SUFFIX(const void*J,const void*M,const void*a,const void*b,const void*as,void*ay,void*ab,int lines,int queries,int steps,int all_states,std::uint64_t stream){return backward<T,S>(J,M,a,b,as,ay,ab,lines,steps,all_states,stream,queries);}
FC_BATCH_TRACE_EXPORT(f32_f32,float,float)
FC_BATCH_TRACE_EXPORT(f32_f64,float,double)
FC_BATCH_TRACE_EXPORT(f64_f32,double,float)
FC_BATCH_TRACE_EXPORT(f64_f64,double,double)

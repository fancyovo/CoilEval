#include "autograd_simd_avx2.h"
#include "simd_avx2.h"
#include <cstddef>

#if !defined(COIL_EVAL_DISABLE_AVX2) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CE_AVX __attribute__((target("avx2,no-fma")))
#define CE_INLINE CE_AVX __attribute__((always_inline)) inline
namespace coil_eval::cpu {
namespace {
template<class T> struct Ops;
template<> struct Ops<float> {
    using V = __m256;
    static constexpr int width = 8;
    static CE_INLINE V set(float x) { return _mm256_set1_ps(x); }
    static CE_INLINE V add(V a,V b) { return _mm256_add_ps(a,b); }
    static CE_INLINE V sub(V a,V b) { return _mm256_sub_ps(a,b); }
    static CE_INLINE V mul(V a,V b) { return _mm256_mul_ps(a,b); }
    static CE_INLINE V div(V a,V b) { return _mm256_div_ps(a,b); }
    static CE_INLINE V sqrt(V a) { return _mm256_sqrt_ps(a); }
    static CE_INLINE V gt(V a,V b) { return _mm256_cmp_ps(a,b,_CMP_GT_OQ); }
    static CE_INLINE V lt(V a,V b) { return _mm256_cmp_ps(a,b,_CMP_LT_OQ); }
    static CE_INLINE V choose(V mask,V yes,V no) { return _mm256_blendv_ps(no,yes,mask); }
    static CE_INLINE V gather(const float* p,int stride) {
        return _mm256_i32gather_ps(p,_mm256_setr_epi32(0,stride,2*stride,3*stride,4*stride,5*stride,6*stride,7*stride),4);
    }
    static CE_INLINE void scatter(float* p,int stride,V value) {
        alignas(32) float lanes[width];_mm256_store_ps(lanes,value);
        for(int k=0;k<width;++k)p[k*stride]=lanes[k];
    }
};
template<> struct Ops<double> {
    using V = __m256d;
    static constexpr int width = 4;
    static CE_INLINE V set(double x) { return _mm256_set1_pd(x); }
    static CE_INLINE V add(V a,V b) { return _mm256_add_pd(a,b); }
    static CE_INLINE V sub(V a,V b) { return _mm256_sub_pd(a,b); }
    static CE_INLINE V mul(V a,V b) { return _mm256_mul_pd(a,b); }
    static CE_INLINE V div(V a,V b) { return _mm256_div_pd(a,b); }
    static CE_INLINE V sqrt(V a) { return _mm256_sqrt_pd(a); }
    static CE_INLINE V gt(V a,V b) { return _mm256_cmp_pd(a,b,_CMP_GT_OQ); }
    static CE_INLINE V lt(V a,V b) { return _mm256_cmp_pd(a,b,_CMP_LT_OQ); }
    static CE_INLINE V choose(V mask,V yes,V no) { return _mm256_blendv_pd(no,yes,mask); }
    static CE_INLINE V gather(const double* p,int stride) {
        return _mm256_i32gather_pd(p,_mm_setr_epi32(0,stride,2*stride,3*stride),8);
    }
    static CE_INLINE void scatter(double* p,int stride,V value) {
        alignas(32) double lanes[width];_mm256_store_pd(lanes,value);
        for(int k=0;k<width;++k)p[k*stride]=lanes[k];
    }
};

template<class T> struct Vec { typename Ops<T>::V x,y,z; };
template<class T> CE_INLINE Vec<T> add(Vec<T> a,Vec<T> b) {
    using O=Ops<T>;return {O::add(a.x,b.x),O::add(a.y,b.y),O::add(a.z,b.z)};
}
template<class T> CE_INLINE Vec<T> mul(Vec<T> a,typename Ops<T>::V b) {
    using O=Ops<T>;return {O::mul(a.x,b),O::mul(a.y,b),O::mul(a.z,b)};
}
template<class T> CE_INLINE typename Ops<T>::V dot(Vec<T> a,Vec<T> b) {
    using O=Ops<T>;return O::add(O::add(O::mul(a.x,b.x),O::mul(a.y,b.y)),O::mul(a.z,b.z));
}
template<class T> CE_INLINE Vec<T> cross(Vec<T> a,Vec<T> b) {
    using O=Ops<T>;return {O::sub(O::mul(a.y,b.z),O::mul(a.z,b.y)),
        O::sub(O::mul(a.z,b.x),O::mul(a.x,b.z)),O::sub(O::mul(a.x,b.y),O::mul(a.y,b.x))};
}
template<class T> CE_INLINE Vec<T> choose(typename Ops<T>::V mask,Vec<T> yes,Vec<T> no) {
    using O=Ops<T>;return {O::choose(mask,yes.x,no.x),O::choose(mask,yes.y,no.y),O::choose(mask,yes.z,no.z)};
}
template<class T> CE_INLINE Vec<T> broadcast(const T* p) {
    using O=Ops<T>;return {O::set(p[0]),O::set(p[1]),O::set(p[2])};
}
template<class T> CE_INLINE Vec<T> gather(const T* p) {
    using O=Ops<T>;return {O::gather(p,3),O::gather(p+1,3),O::gather(p+2,3)};
}
template<class T> CE_INLINE void scatter(T* p,Vec<T> a) {
    using O=Ops<T>;O::scatter(p,3,a.x);O::scatter(p+1,3,a.y);O::scatter(p+2,3,a.z);
}

template<class T,bool Potential,bool Gradient> CE_INLINE void pair_reverse(
    Vec<T> r,Vec<T> w,Vec<T> a,const typename Ops<T>::V* h,Vec<T>& dw,Vec<T>& dr) {
    using O=Ops<T>;using V=typename O::V;
    const V zero=O::set(0),one=O::set(1),minus_three=O::set(-3),epsilon=O::set(T(1e-24));
    const Vec<T> zero3{zero,zero,zero};
    const V distance2=dot(r,r);
    // std::max(distance2,epsilon) retains a NaN first argument; AVX max does not.
    const V r2=O::choose(O::lt(distance2,epsilon),epsilon,distance2);
    const V inv=O::div(one,O::sqrt(r2)),f3=O::mul(O::mul(inv,inv),inv);
    const V active=O::gt(distance2,epsilon);
    if constexpr(Potential) {
        dw=mul(a,inv);
        dr=choose(active,mul(r,O::mul(O::mul(O::set(-1),dot(a,w)),f3)),zero3);
    } else {
        const Vec<T> t=cross(w,r);
        Vec<T> hr=zero3,ht=zero3,hw=zero3;V scalar=zero;
        if constexpr(Gradient) {
            hr={O::add(O::add(O::mul(h[0],r.x),O::mul(h[1],r.y)),O::mul(h[2],r.z)),
                O::add(O::add(O::mul(h[3],r.x),O::mul(h[4],r.y)),O::mul(h[5],r.z)),
                O::add(O::add(O::mul(h[6],r.x),O::mul(h[7],r.y)),O::mul(h[8],r.z))};
            ht={O::add(O::add(O::mul(h[0],t.x),O::mul(h[3],t.y)),O::mul(h[6],t.z)),
                O::add(O::add(O::mul(h[1],t.x),O::mul(h[4],t.y)),O::mul(h[7],t.z)),
                O::add(O::add(O::mul(h[2],t.x),O::mul(h[5],t.y)),O::mul(h[8],t.z))};
            const Vec<T> axes[3]={{one,zero,zero},{zero,one,zero},{zero,zero,one}};
            for(int j=0;j<3;++j) {
                const Vec<T> column{h[j],h[3+j],h[6+j]};
                scalar=O::add(scalar,dot(column,cross(w,axes[j])));
                hw=add(hw,cross(axes[j],column));
            }
        }
        const V f5=O::div(f3,r2);
        const Vec<T> at=choose(active,add(mul(a,f3),mul(hr,O::mul(minus_three,f5))),mul(a,f3));
        dw=add(cross(r,at),mul(hw,f3));dr=cross(at,w);
        Vec<T> radial=add(dr,mul(ht,O::mul(minus_three,f5)));
        const V factor=O::add(
            O::mul(O::mul(minus_three,f5),O::add(dot(a,t),scalar)),
            O::mul(O::div(O::mul(O::set(15),f5),r2),dot(t,hr)));
        radial=add(radial,mul(r,factor));
        dr=choose(active,radial,dr);
    }
    const V coefficient=O::set(T(1e-7));
    dw=mul(dw,coefficient);dr=mul(dr,coefficient);
}

template<class T,bool Potential,bool Segment,bool Gradient> CE_AVX void block(
    const VjpBuffers& buffers,int first) {
    using O=Ops<T>;using V=typename O::V;
    const auto *p=static_cast<const T*>(buffers.p),*w=static_cast<const T*>(buffers.w),
               *q=static_cast<const T*>(buffers.q),*a=static_cast<const T*>(buffers.a),
               *h=static_cast<const T*>(buffers.h);
    auto *dp=static_cast<T*>(buffers.dp),*dw=static_cast<T*>(buffers.dw),*dq=static_cast<T*>(buffers.dq);
    const V zero=O::set(0),minus_one=O::set(-1);
    const Vec<T> zero3{zero,zero,zero};
    Vec<T> ar=zero3,aw=zero3;
    Vec<T> position,element,point,adjoint;V matrix[9];
    if constexpr(Segment) {
        position=gather(p+3*first);element=gather(w+3*first);
    } else {
        point=gather(q+3*first);adjoint=a?gather(a+3*first):zero3;
        if constexpr(Gradient)for(int j=0;j<9;++j)matrix[j]=O::gather(h+9*first+j,9);
    }
    const int count=Segment?buffers.points:buffers.segments;
    for(int inner=0;inner<count;++inner) {
        if constexpr(Segment) {
            point=broadcast(q+3*inner);adjoint=a?broadcast(a+3*inner):zero3;
            if constexpr(Gradient)for(int j=0;j<9;++j)matrix[j]=O::set(h[9*inner+j]);
        } else {
            position=broadcast(p+3*inner);element=broadcast(w+3*inner);
        }
        Vec<T> vr,vw;
        pair_reverse<T,Potential,Gradient>(add(point,mul(position,minus_one)),element,adjoint,matrix,vw,vr);
        ar=add(ar,vr);
        if constexpr(Segment)aw=add(aw,vw);
    }
    if constexpr(Segment) {
        if(dp)scatter(dp+3*first,mul(ar,minus_one));
        if(dw)scatter(dw+3*first,aw);
    } else scatter(dq+3*first,ar);
}

template<class T,bool Potential,bool Segment,bool Gradient> int run(const VjpBuffers& buffers) {
    const int count=Segment?buffers.segments:buffers.points;
    const int groups=count/Ops<T>::width;
    #pragma omp parallel for num_threads(buffers.threads) schedule(static)
    for(int group=0;group<groups;++group)
        block<T,Potential,Segment,Gradient>(buffers,group*Ops<T>::width);
    return groups*Ops<T>::width;
}

template<class T,bool Segment> int select(const VjpBuffers& buffers,bool potential) {
    if(potential)return run<T,true,Segment,false>(buffers);
    if(buffers.h)return run<T,false,Segment,true>(buffers);
    return run<T,false,Segment,false>(buffers);
}
template<class T> int dispatch(const VjpBuffers& buffers,bool potential,bool segment_outputs) {
    if(!avx2_available())return 0;
    return segment_outputs?select<T,true>(buffers,potential):select<T,false>(buffers,potential);
}
}
int autograd_vjp_f32(const VjpBuffers& buffers,bool potential,bool segment_outputs) {
    return dispatch<float>(buffers,potential,segment_outputs);
}
int autograd_vjp_f64(const VjpBuffers& buffers,bool potential,bool segment_outputs) {
    return dispatch<double>(buffers,potential,segment_outputs);
}
}
#else
namespace coil_eval::cpu {
int autograd_vjp_f32(const VjpBuffers&,bool,bool) {return 0;}
int autograd_vjp_f64(const VjpBuffers&,bool,bool) {return 0;}
}
#endif

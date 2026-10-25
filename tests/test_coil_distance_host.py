"""The exact sampled distance index preserves the original active-pair rule."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope='module')
def distance_search(tmp_path_factory):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++17 compiler unavailable')
    folder=tmp_path_factory.mktemp('coil-distance')
    header=Path(__file__).resolve().parents[1]/'native/include/pipeline/coil_distance.cuh'
    source=folder/'search.cpp'
    source.write_text(f'#include "{header.as_posix()}"\n'+r'''
struct P { struct V { double x,y,z; } point; int label; };
extern "C" int compare_search(const double*xyz,const int*labels,int count,double*answer) {
 std::vector<P> points;
 for(int i=0;i<count;++i)points.push_back({{xyz[3*i],xyz[3*i+1],xyz[3*i+2]},labels[i]});
 using namespace coil_eval::pipeline::geometry_detail;
 DistanceMinimum reference;
 for(size_t i=0;i<points.size();++i)for(size_t j=i+1;j<points.size();++j) {
  if(points[i].label==points[j].label)continue;
  double x=points[i].point.x-points[j].point.x,y=points[i].point.y-points[j].point.y,
         z=points[i].point.z-points[j].point.z;
  double d=std::sqrt(x*x+y*y+z*z);
  if(d<reference.distance)reference={d,i,j};
 }
 const auto result=closest_distinct_label_pair(points);
 answer[0]=result.distance;answer[1]=double(result.left);answer[2]=double(result.right);
 return result.distance==reference.distance && result.left==reference.left && result.right==reference.right;
}
''',encoding='utf-8')
    library=folder/('search.dll' if sys.platform=='win32' else 'search.so')
    cmd=[compiler,'-std=c++17','-O3','-ffp-contract=off','-shared',str(source),'-o',str(library)]
    if sys.platform!='win32':cmd.insert(1,'-fPIC')
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    runtime=(os.add_dll_directory(str(Path(compiler).resolve().parent)) if sys.platform=='win32' else None)
    loaded=ct.CDLL(str(library));fun=loaded.compare_search
    p=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
    fun.argtypes=[p,ip,ct.c_int,p];fun.restype=ct.c_int
    def check(points,labels):
        xyz=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
        labels=np.ascontiguousarray(labels,dtype=np.int32)
        answer=np.empty(3)
        ok=fun(xyz.ctypes.data_as(p),labels.ctypes.data_as(ip),len(labels),answer.ctypes.data_as(p))
        assert ok,answer
        return answer
    yield check
    if runtime is not None:runtime.close()


@pytest.mark.parametrize('count',[0,1,2,17,64,511,1000])
def test_random_points_and_labels(distance_search,count):
    rng=np.random.default_rng(20261025+count)
    for _ in range(4):
        points=rng.normal(size=(count,3))
        labels=rng.integers(0,32,count)
        distance_search(points,labels)


@pytest.mark.parametrize('scale',[1e-200,1e-100,1.,1e100,1e154])
def test_ranges_and_nearly_coincident_points(distance_search,scale):
    rng=np.random.default_rng(20261025)
    points=rng.normal(size=(96,3))*scale
    labels=np.arange(96)%8
    points[43]=np.nextafter(points[7],np.inf)
    distance_search(points,labels)


def test_rounded_distance_tie_and_global_first_pair(distance_search):
    points=np.zeros((48,3));points[:,0]=np.arange(48)+10
    points[0]=[0,0,0];points[1]=[1,1e-9,0];points[2]=[1,0,0]
    labels=np.zeros(48,dtype=int);labels[1:3]=1
    answer=distance_search(points,labels)
    np.testing.assert_array_equal(answer,[1,0,1])


def test_zero_distance_and_all_one_label(distance_search):
    points=np.zeros((64,3));labels=np.arange(64)%8
    np.testing.assert_array_equal(distance_search(points,labels),[0,0,1])
    answer=distance_search(points,np.zeros(64,dtype=int))
    assert np.isinf(answer[0]) and tuple(answer[1:])==(0,0)


def test_nonfinite_fallback_retains_nested_loop_result(distance_search):
    points=np.zeros((40,3));points[:,0]=np.arange(40)
    points[0,0]=np.nan;points[4,1]=np.inf
    distance_search(points,np.arange(40)%4)

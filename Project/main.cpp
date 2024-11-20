#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <iostream>
#include <cmath>

#include "dummy.h"
#include "init_test.h"


void print_iter(double* a, int size){
    int i;
    for (i=0; i<size; i++){
        std::cout << a[i]<<" ";
    }
    std::cout << std::endl;
}


int main() {
    using namespace boost;

    // 그래프 자료구조 정의
    typedef adjacency_list<vecS, vecS, undirectedS> Graph;

    // 노드에 해당하는 구간 정의
    std::vector<double> intervals = {1.6, 2.4, 2.5, 4.8, 5.2}; // [1,2], [2.5,3.5], ...
    
    /* 
    
    Graph g(intervals.size());

    // 간선 추가 (두 구간이 겹치면 간선 생성)
    for (size_t i = 0; i < intervals.size(); ++i) {
        for (size_t j = i + 1; j < intervals.size(); ++j) {
            if (intervals[i] <= intervals[j] + 1.0 && intervals[j] <= intervals[i] + 1.0) {
                add_edge(i, j, g); // 노드 i와 j를 연결
            }
        }
    }

    // 출력: 간선 정보
    std::cout << "Edges in the graph:" << std::endl;
    for (auto [ei, ei_end] = edges(g); ei != ei_end; ++ei) {
        std::cout << source(*ei, g) << " -- " << target(*ei, g) << std::endl;
    }

    */

   // m 구하기
    double leftmost = *(intervals.begin());
    int m = 1;
   // std::cout << leftmost << std::endl;
    for (auto interval : intervals){
        if (interval > leftmost + 1){
            leftmost = interval;
            m++;
        }
    }
 
    // Ii의 첫 번째 원소(decomposition[i])와 Ii의 size(count[i]) 구하기 
    double* decomposition = new double[m];
    double* count = new double[m];
    int i = 0;
    double elem = *(intervals.begin());
    decomposition[0] = elem;
    for (auto interval : intervals){
        std::cout<< interval << " " << elem <<std::endl;
        if (interval > elem + 1.0){
            elem = interval;
            decomposition[++i] = elem;
            count[i] = 1;
        }
        else{
            count[i]++;
        }
    }

    std::cout << m << std::endl;

    int j;
    for (j=0; j<m; j++){
        std::cout << decomposition[j]<<" ";
    }
    std::cout << std::endl;

    for (j=0; j<m; j++){
        std::cout << count[j]<<" ";
    }
    std::cout << std::endl;

    



    



    return 0;
}



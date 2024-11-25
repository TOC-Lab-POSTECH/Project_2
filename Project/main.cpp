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
    std::vector<double> intervals = {1.0, 1.1, 1.3, 2.01, 2.15, 2.2, 2.4, 3.02, 3.16, 3.3}; // [1,2], [2.5,3.5], ...
    
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

    std::cout << "m is " << m << std::endl;
 
    // Ii의 첫 번째 원소(decomposition[i])와 Ii의 size(count[i]) 구하기 
    // double* decomposition = new double[m];
    std::vector<double> decomposition(m);
    // int* count = new int[m];
    std::vector<int> count(m);
    int i = 0;
    double elem = *(intervals.begin());
    decomposition[0] = elem;
    for (auto interval : intervals){
        // std::cout<< interval << " " << elem <<std::endl;
        if (interval > elem + 1.0){
            elem = interval;
            decomposition[++i] = elem;
            count[i] = 1;
        }
        else{
            count[i]++;
        }
    }
    i = 0;
    for (auto size: count){
        std::cout << "I" << ++i << " size: " << size << std::endl;
    }
    i=0;
    for (auto interval: decomposition){
        std::cout << i++ << "th interval: " << interval << std::endl;
    }

    
    // 원하는 interval 위치를 쉽게 접근하기 위해 만든 변수
    // int* positions = new int[m];
    std::vector<int> positions(m);
    for (i = 1; i < m; i++){
        positions[i] += positions[i-1] + count[i-1];
    }

    i = 0;
    for (auto position: positions){
        std::cout << i++ << "th interval position: " << position << std::endl;
    }

// -------------------------------------------------------------------------------------
// 여기까지는 문제 없음!
    
    
    /*
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
    */
    

    int j;
    // int* s = new int[intervals.size()]; 
    std::vector<int> s(intervals.size()); // smallest set의 size
    //int* j_prime = new int[intervals.size()]; 
    std::vector<int> j_prime(intervals.size()); // smallest set을 만드는 j'
    // base case(i = 1), s[1,j] = |A_{1,j}|
    for(j = 0; j < count[0]; j++){
        s[j] = j;
    }

    // i >= 2에서 s[i,j]를 구하는 과정
    for(i = 1; i < m; i++){
        int* k = new int[count[i]];
        int ii = 0;
        std::cout<< "s[" << i << ", j] finding" << std::endl;
        for(j = 0; j < count[i]; j++){
        // k(j) 계산하기
            while(intervals[positions[i-1] + ii] + 1.0 < intervals[positions[i] + j]){
                std::cout << ii<<"th interval "<< intervals[positions[i-1]+ii] << " is disjoint with " << intervals[positions[i]+j] << std::endl;
                ii++;
            }
            k[j] = ii-1;
            std::cout<< "   " << j <<"th k: " << k[j] << std::endl;
        }


        //-----------------------------------------------------------------------------------------------------------------------------------------
        // k 구하기까지는 문제 없음!
/*
        // j가 1일 때 s[i+1, 1] = min(s[i,j'] + |A_{i,j',1}|)
        // j'이 1일 때부터 하나씩 계산
        int min = s[positions[i-1]] + k[0];
        int min_j_prime = 0;
        for(int jj = 0; jj < k[0]; jj++){
            int comp = s[positions[i-1] + jj] + k[0] - jj;
            if(comp < min){
                min = comp;
                min_j_prime = jj;
            }
        }
        s[positions[i]] = min;
        j_prime[positions[i]] = min_j_prime;
        
        
        // j가 2 이상일 때
        for(j = 1; j < count[i]; j++){
            // 우선 j-1에서의 j'을 최솟값으로 설정
            min = s[positions[i] + j-1] + k[j] - k[j-1] + 1;
            min_j_prime = j_prime[positions[i] + j-1];
            for(int jj = k[j-1] + 1; jj <= k[j]; jj++){
                // j' > k(j-1)인 경우의 s[i,j']+|A_{i,j',j}|와 각각 비교
                int comp = s[positions[i-1]+jj] + k[j]-jj;
                if(comp < min){
                    min = comp;
                    min_j_prime = jj;
                }
            }
            s[positions[i] + j] = min;
            j_prime[positions[i] + j] = min_j_prime;
        }
        delete k;
    }

    // 최종 minimum set을 구하는 과정 
    int min = s[positions[m-1]] + count[m-1] - 1;
    int min_j_prime = 0;
    for(j = 1; j < count[m-1]; j++){
        int comp = s[positions[m-1] + j] + count[m-1] - j - 1;
        if(comp < min){
            min = comp;
            min_j_prime = j;
        }
        std::cout << "min_j_prime: "<< min_j_prime<< std::endl;
    }
    
    // minimum set S를 구하기 위한 j' index들의 집합.
    //int* j_index = new int[m];
    std::vector<int> j_index(m);

    j_index[m-1] = min_j_prime;
    for (i = m-2; i >= 0; i--){
        j_index[i] = j_prime[positions[i+1] + j_index[i+1]];
    }

    //std::vector<int> j_prime_index(j_index, j_index + m*sizeof(int));

    for(auto& v: j_index){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    std::vector<double> S; // minimum set S 구성

    // A_{1,j_{1}} 넣기
    for(j = 0; j < j_index[0]; j++){
        S.push_back(intervals[j]);
    }

    // A_{i,j_{i},j_{i+1}} 넣기
    for(i = 1; i < m; i++){
        // A_{i,j_{i},j_{i+1}} 중 앞부분(I_{i}의 원소들)
        j = j_index[i-1] + 1;
        while(intervals[positions[i-1] + j] + 1.0 < intervals[positions[i] + j_index[i]]){
            S.push_back(intervals[positions[i-1] + j]);
            j++;
        }
        // A_{i,j_{i},j_{i+1}} 중 뒷부분(I_{i+1}의 원소들)
        j = j_index[i] - 1;
        while(intervals[positions[i-1] + j_index[i-1]] + 1.0 < intervals[positions[i] + j]){
            S.push_back(intervals[positions[i] + j]);
            j--;
        }
    }

    // {I_{m,k} in I_{m}: j < k} 넣기
    for(j = j_index[m-1] + 1; j < count[m-1]; j++){
        S.push_back(intervals[positions[m-1] + j]);
    }

    for(auto& I : S) {
        std::cout << I << " ";
    }
    std::cout << std::endl;
    
   // delete j_index;
    //delete s;
    //delete j_prime;
    //delete decomposition;
    //delete count;
    //delete positions;

*/
    return 0;
}



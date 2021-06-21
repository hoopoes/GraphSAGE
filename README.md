# GraphSAGE Implementation with User-Item Setting  
----

<center><img src="/image/main_img.JPG" width="70%"></center>  

## Explanation  
Base Github: [Stellargraph](https://github.com/stellargraph/stellargraph)  
Original Paper: [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)  
Paper Review Article written in Korean: [Review Article](https://greeksharifa.github.io/machine_learning/2020/12/31/Graph-Sage/)  

본 코드는 GraphSAGE 알고리즘을 Tensorflow로 구현한 코드이다.

GraphSAGE 논문 원본은 1개의 Node Type을 가진 Homogenous Graph에 대해서만 설명하고 있다.  
그런데 추천 시스템에서는 일반적으로 User-Item 설정을 두기 때문에, 우리에게는 Bipartite, 즉 이분 그래프 구조가 필요하다.  

그리고 이러한 설정에서는 보통 User와 Item은 전혀 다른 Feature를 갖게 된다.  
따라서 User와 Item은 다른 Weight Matrices에 의해 분리되어 학습되어야 한다.  

본 Repository는 **Stellagraph**의 Component를 상당 부분 그대로 가져다 쓴 상태로 구현되어 있다.  
다만 직접적으로 이 Library를 Import할 필요 없게 수정되어 있기 때문에, 기본적인 Bipartite GraphSAGE의 구조에 대해 쉽게 살펴볼 수 있는 코드라고 생각하면 될 것이다.  

대용량의 Bipartite Graph 구조의 데이터에 기반한 GNN 추천 시스템은 [이 곳](https://github.com/ocasoyy/Bipartite-Graph-Isomorphism-Network)에서 구현하도록 할 것이다.  


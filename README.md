# PairSAGE
----

Sample Image  
<center><img src="/image/image01.png" width="40%"></center>  

## Overview  
Author: Youyoung Jang  
Base Algorithm: GraphSAGE  
Base Github: [Stellargraph](https://github.com/stellargraph/stellargraph)  
Original Paper: [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)  
Paper Review Article written in Korean: [Review Article](https://greeksharifa.github.io/machine_learning/2020/12/31/Graph-Sage/)  

This Algorithm is based on GraphSAGE Algorithm.  
Originally GraphSAGE is for homogenenous graph which has only one type node.  
When building a Recommendation System, we usually encounter the bipartite graph. This bipartite graph is composed of User-Item pair setting and each node has unique feature and characteristic.  
So User node and Item node must be trained separately with different weight matrices.  

**Pair** in PairSAGE is inserted to stress the importance of **Pair** relationship between User and Item.  
This project can be used for common recommendation system. If features are diverse and new nodes are added frequently this project might be helpful to you.  

----



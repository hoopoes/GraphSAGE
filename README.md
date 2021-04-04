# GraphSAGE Implementation with User-Item Setting  
----

<center><img src="/image/main_img.JPG" width="70%"></center>  

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

This Algorithm has been implemented  to stress the importance of **Pair** relationship between User and Item.  
This project can be used for common recommendation system. If features are diverse and new nodes are being added frequently, then this code might be helpful to you.  

**Stellagraph** is a descent graph neural network library and has been really helpful to understand the process of GraphSAGE algorithm. However if you are only interested in Bipartite User-Item Recommendation, this library can be unnecesarily complex to you.  

So I present you with simpler code which makes you do not have to import Stellargraph. Note that many modules are just same as Stellargraph but modified a little bit.  

----

## Structure  

**Base Components**  
1) Graph Object  
2) Breadth First Walker  
3) Bipartite Link Generator  
4) GraphSAGE Model  

**Relationship among Modules**  





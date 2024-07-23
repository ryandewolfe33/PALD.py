# PALD.py

This is a pure python implementation of the PALD clustering algorithm presented in *A social perspective on perceived distances reveals deep community structure*. The official implemnetation is an R package available at https://github.com/moorekatherine/pald.

# APALD.py

This is a custom work in progress on Approximate PArtitioned Local Depth for clustering. It's showing some promise.  

## WARNING
This is by no means an efficient implementation, and the algrothim runs in O(n^3) with the number of points. Usage is not recomended for more than a few hundred points.

Test data was taken from https://github.com/moorekatherine/pald/tree/main/data and converted to csv files (after dropping headers) using https://github.com/vnmabus/rdata. 

Aggregation.txt data is a Benchmark data set Aggregation consisting of n = 788 points.
Link to data: http://cs.joensuu.fi/sipu/datasets/Aggregation.txt
A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.


Included in "cognate.txt"
I. Dyen, J. B. Kruskal, P. Black, An Indoeuropean classification: A lexicostatistical experiment.  Trans. Am. Phil. Soc.82, iii-132 (1992)
Downloaded from: https://github.com/moorekatherine/pald-communities
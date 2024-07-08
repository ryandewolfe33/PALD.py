# PALD.py

This is a pure python implementation of the PALD clustering algorithm presented in *A social perspective on perceived distances reveals deep community structure*. The official implemnetation is an R package available at https://github.com/moorekatherine/pald. 

## WARNING
This is by no means an efficient implementation, and the algrothim runs in O(n^3) with the number of points. Usage is not recomended for more than a few hundred points.

Test data was taken from https://github.com/moorekatherine/pald/tree/main/data and converted to csv files (after dropping headers) using https://github.com/vnmabus/rdata. 
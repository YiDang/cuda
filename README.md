# cuda
use following command to compile:
nvcc part2.cu -lcublas_static -lculibos -o part2
then use following to run:
./part2

I have defined M_ N_ P_ at the beginning which indicate the size of the input matrix:
A(M_ * N_) B(N_ * P_)
These numbers can be changed as you like.

In the main() function, you can set showma to 1 if you want to observe the result matrix of each mathod(better set it to 0 when matrix is large).
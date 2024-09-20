#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
using namespace std;

const unsigned int NUM_ROWS_A = 1000;
const unsigned int NUM_COLS_A = 1000;
const unsigned int NUM_ROWS_B = NUM_COLS_A;
const unsigned int NUM_COLS_B = 1000;
const unsigned int thread_count = 4;

std::mutex result_mutex;

void parallel_helper(int** Matrix_A, int** Matrix_B, int** Result,
    unsigned int num_cols_a, unsigned int num_cols_b,
    unsigned int row_start, unsigned int row_end);


void InitArray(int**& Matrix, unsigned int Rows, unsigned int Cols) {
    Matrix = new int* [Rows];
    if (Matrix == NULL) {
        exit(2);
    }
    for (unsigned int i = 0; i < Rows; i++) {
        Matrix[i] = new int[Cols];
        if (Matrix[i] == NULL) {
            exit(2);
        }
        for (unsigned int j = 0; j < Cols; j++) {
            Matrix[i][j] = rand() % 100 + 1;
        }
    }
}

void DisplayArray(int**& Matrix, unsigned int Rows, unsigned int Cols) {
    for (unsigned int i = 0; i < Rows; i++) {
        for (unsigned int j = 0; j < Cols; j++) {
            cout << " [" << Matrix[i][j] << "] ";
        }
        cout << endl;
    }
}

//Sequential Matrix Multiplication
void sequential_matrix_multiply(int**& Matrix_A, unsigned int num_rows_a, unsigned int num_cols_a,
    int**& Matrix_B, unsigned int num_rows_b, unsigned int num_cols_b,
    int**& Result) {
    for (unsigned int i = 0; i < num_rows_a; i++) {
        for (unsigned int j = 0; j < num_cols_b; j++) {
            Result[i][j] = 0;
            for (unsigned int k = 0; k < num_cols_a; k++) {
                Result[i][j] += Matrix_A[i][k] * Matrix_B[k][j];
            }
        }
    }
}

// Responsible for breaking the data into chunks for parallel processing
void parallel_matrix_multiply(int**& Matrix_A, unsigned int num_rows_a, unsigned int num_cols_a,
    int**& Matrix_B, unsigned int num_rows_b, unsigned int num_cols_b,
    int**& Result, unsigned int thread_count) {

    std::vector<std::thread> multiplication_threads;

    // Find out how many rows each thread will multiply
    unsigned int rows_per_thread = num_rows_a / thread_count;

    // Calculate the number of extra rows if there are any
    unsigned int extra_rows = num_rows_a % thread_count;
    unsigned int row_start = 0;

    // Spawn threads
    for (unsigned int i = 0; i < thread_count; i++) {
        // Find out where each thread ends and the next begins
        unsigned int row_end = row_start + rows_per_thread;
        // Last thread handles extra rows
        if (i == thread_count - 1)
            row_end += extra_rows;

        // Spawn thread
        multiplication_threads.push_back(std::thread(parallel_helper, Matrix_A, Matrix_B, Result, num_cols_a, num_cols_b, row_start, row_end));
        
        // Update start point for next thread
        row_start = row_end;
    }

    for (auto& thread : multiplication_threads) thread.join();
}


// Thread logic for multiplying chunks of the matrix
void parallel_helper(int** Matrix_A, int** Matrix_B, int** Result,
    unsigned int num_cols_a, unsigned int num_cols_b,
    unsigned int row_start, unsigned int row_end) {
   
    for (unsigned int i = row_start; i < row_end; i++) {
        for (unsigned int j = 0; j < num_cols_b; j++) {
            int temp = 0;
            for (unsigned int k = 0; k < num_cols_a; k++) {
                temp += Matrix_A[i][k] * Matrix_B[k][j];
            }

            result_mutex.lock();
            Result[i][j] = temp;
            result_mutex.unlock();
        }
    }
}

int main()
{
    int** MatrixA = nullptr;
    int** MatrixB = nullptr;
    int** Result = nullptr;

    //Allocate data for the resulting arrays
    Result = new int* [NUM_ROWS_A];
    for (unsigned int i = 0; i < NUM_ROWS_A; i++) {
        Result[i] = new int[NUM_COLS_B];
    }

    InitArray(MatrixA, NUM_ROWS_A, NUM_COLS_A);
    InitArray(MatrixB, NUM_ROWS_B, NUM_COLS_B);

    cout << "Evaluating Sequential Task" << endl;

    chrono::duration<double> Seq_Time(0);
    auto startTime = chrono::high_resolution_clock::now();
    sequential_matrix_multiply(MatrixA, NUM_ROWS_A, NUM_COLS_A,
        MatrixB, NUM_ROWS_B, NUM_COLS_B,
        Result);

    Seq_Time = chrono::high_resolution_clock::now() - startTime;

    cout << "FINAL RESULTS" << endl;
    cout << "=============" << endl;
    cout << "Sequential Processing took: " << Seq_Time.count() * 1000 << endl;

    cout << endl <<"Evaluating Parallel Task" << endl;

    chrono::duration<double> Par_Time(0);
    startTime = chrono::high_resolution_clock::now();
    parallel_matrix_multiply(MatrixA, NUM_ROWS_A, NUM_COLS_A, MatrixB, NUM_ROWS_B, NUM_COLS_B, Result, thread_count);
    Par_Time = chrono::high_resolution_clock::now() - startTime;

    cout << "FINAL RESULTS" << endl;
    cout << "=============" << endl;
    cout << "Parallel Processing took: " << Par_Time.count() * 1000 << endl;

    auto efficiency = (Seq_Time / Par_Time) / thread_count;

    cout << "Efficiency: " << efficiency << endl;
    
    //DisplayArray(MatrixA, NUM_ROWS_A, NUM_COLS_A);
    //cout << endl << endl;
    //DisplayArray(MatrixB, NUM_ROWS_B, NUM_COLS_B);
    //cout << endl << endl;
    //DisplayArray(Result, NUM_ROWS_A, NUM_COLS_B);

    return 1;
}
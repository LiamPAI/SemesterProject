import torch
import sys
sys.path.append("/Users/liamcurtis/Documents/ETH Classes/Winter 2024/Semester Project/Sem_Project_LC/cmake-build-debug/python")
import metal_foams


if __name__ == "__main__":
    params = metal_foams.GenerationParams()
    params.datasetSize = 100
    params.numBranches = 5
    print(params.datasetSize)

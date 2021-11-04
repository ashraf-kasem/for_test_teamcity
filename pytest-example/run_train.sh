docker run \
  -v "$(pwd)"/logs:/pytest-example/logs \
  --privileged \
  pytest-example \
  python3 Train_complex_DataSet_with_CNN.py
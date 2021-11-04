docker run \
  -v "$(pwd)"/logs:/pytest-example/logs \
  -p 6006:6006/tcp \
  pytest-example \
  tensorboard --bind_all --logdir="./logs"
  
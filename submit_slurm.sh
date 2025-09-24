export http_proxy=http://127.0.0.1:11000
export https_proxy=http://127.0.0.1:11000


git config --global http.proxy http://127.0.0.1:11000
git config --global https.proxy http://127.0.0.1:11000

srun -p def -n 1 -c 80 --pty /bin/bash